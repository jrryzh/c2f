import os
import cv2
import time
import random
import argparse
import numpy as np
from shutil import copyfile
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataloader_transformer import load_dataset
from utils.logger import setup_logger
from utils.utils import Config, Progbar, to_cuda

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--seed', type=int, default=42) 
    # path
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="../check_points", )
    # training
    parser.add_argument('--Image_W', type=int, default=256)
    parser.add_argument('--Image_H', type=int, default=256)
    parser.add_argument('--patch_W', type=int, default=256)
    parser.add_argument('--patch_H', type=int, default=256)
    
    # dataset 
    parser.add_argument('--dataset', type=str, default="MOViD_A", help = "select dataset")
    parser.add_argument('--data_type', type=str, default="image", help = "select image or video model")
    parser.add_argument('--batch', type=int, default=1)

    # model
    parser.add_argument('--model', type=str, default="original", help = "select model type")

    args = parser.parse_args()

    if args.data_type=="image":
        if args.model == "original":
            from src.image_model import C2F_Seg
        elif args.model == "rgbd_6channel":
            from src.image_model_depth_4channel import C2F_Seg
        elif args.model == "rgbd_fusion":
            from src.image_model_depth_fusion import C2F_Seg
        elif args.model == "depth_only":
            from src.image_model_depth import C2F_Seg
        elif args.model == "rgbd_linearfusion":
            from src.image_model_depth_linearfusion import C2F_Seg
    elif args.data_type=="video":
        from src.video_model import C2F_Seg

    torch.cuda.set_device(0)

    args.path = os.path.join(args.check_point_path, args.path)
    os.makedirs(args.path, exist_ok=True)

    config_path = os.path.join(args.path, 'c2f_seg_{}.yml'.format(args.dataset))
    if not os.path.exists(config_path):
        copyfile('./configs/c2f_seg_{}.yml'.format(args.dataset), config_path)
    
    # load config file
    config = Config(config_path)
    config.path = args.path
    config.batch_size = args.batch
    config.dataset = args.dataset

    log_file = 'log-{}.txt'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)
    
    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./configs/c2f_seg_{}.yml'.format(args.dataset), config_path)

    # save samples and val results
    # os.makedirs(os.path.join(args.path, 'train_samples'), exist_ok=True)
    os.makedirs(os.path.join(args.path, 'val_samples'), exist_ok=True)

    for k in config._dict:
        logger.info("{}:{}".format(k, config._dict[k]))

    # init device
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.device = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    model = C2F_Seg(config, mode='train', logger=logger)
    model.load(is_test=False, prefix = config.stage2_iteration)
    model.to(config.device)
    
    # load dataset
    train_dataset, test_dataset = load_dataset(config, args, "train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_num_workers,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.test_num_workers,
        drop_last=False,
    )
    
    sample_iterator = test_dataset.create_iterator(config.sample_size)
    steps_per_epoch = len(train_dataset) // config.batch_size
    iteration = model.iteration
    sample_iter = model.sample_iter

    epoch = model.iteration // steps_per_epoch
    
    logger.info('Start from epoch:{}, iteration:{}'.format(epoch, iteration))
    writer = SummaryWriter(args.path)

    model.train()
    keep_training = True
    best_score = {}
    # lm rate
    if config.lm_rate > 0:
        lm_size = int(config.batch_size * config.lm_rate)
        mc = [1] * lm_size + [0] * (config.batch_size - lm_size)
    else:
        mc = None
    while (keep_training):
        epoch += 1
        stateful_metrics = ['epoch', 'iter', 'lr']
        progbar = Progbar(len(train_dataset), max_iters=steps_per_epoch,
                        width=20, stateful_metrics=stateful_metrics)
        
        for items in train_loader:
            model.train()
            items = to_cuda(items, config.device)
            items['mc'] = None
            counts = items['counts'].squeeze().bool().sum()
            if counts==0:
                continue
            r_loss, logs = model.get_losses(items)
            model.backward(r_loss)
            torch.cuda.empty_cache()
            iteration = model.iteration
            sample_iter = model.sample_iter

            writer.add_scalar("loss/r_loss", r_loss, model.iteration)
            logs = [("epoch", epoch), ("iter", iteration), ('lr', model.sche.get_lr()[0])] + logs
            progbar.add(config.batch_size, values=logs)

            if iteration % config.val_vis_iters == 0:
                model.eval()
                # For image amodal dataset
                # if args.dataset in ["KINS", "COCOA"]:
                if args.data_type=="image":
                    with torch.no_grad():
                        items = next(sample_iterator)
                        items = to_cuda(items, config.device)
                        img_id = items["img_id"]
                        anno_id = items["anno_id"]
                        sample_iter = model.sample_iter
                        loss_eval = model.batch_predict_maskgit(items, sample_iter, 'val')
                        iou = loss_eval['iou'].item() / (loss_eval['iou_count'].item() + 1e-7)
                        invisible_iou_ = loss_eval['invisible_iou_'].item()
                        print(
                            "img_id: ",int(anno_id[0].cpu().detach().numpy())," - ",
                            "anno_id: ",int(anno_id[0].cpu().detach().numpy())," - ",
                            "Iou: ", iou, " - ",
                            "in-Iou", invisible_iou_, 
                        )
                # For video amodal dataset
                elif args.data_type=="video":
                # elif args.dataset in ["Fishbowl", "MOViD_A"]:
                    with torch.no_grad():
                        items = next(sample_iterator)
                        items = to_cuda(items, config.device)
                        video_ids, object_ids = items["video_ids"][0][0], items["object_ids"][0][0]
                        sample_iter = model.sample_iter
                        loss_eval = model.batch_predict_maskgit(items, sample_iter, 'val')
                        iou = loss_eval['iou_post'].item()
                        invisible_iou_ = loss_eval['invisible_iou_'].item()
                        print(
                            "video_ids: ",int(video_ids.cpu().detach().numpy())," - ",
                            "object_ids: ",int(object_ids.cpu().detach().numpy())," - ",
                            "Iou: ", iou, " - ",
                            "in-Iou", invisible_iou_, 
                        )
            if iteration % config.log_iters == 0:
                logger.debug(str(logs))
            if iteration % config.save_iters == 0:
                model.save(prefix='{}'.format(iteration))
            if iteration >= config.max_iters:
                keep_training = False
                break

