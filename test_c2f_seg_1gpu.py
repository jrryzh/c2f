import os
import cv2
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import torch
from torch.utils.data import DataLoader
from data.dataloader_transformer import load_dataset
from utils.logger import setup_logger
from utils.utils import Config, to_cuda

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="../check_points")
    parser.add_argument('--vq_path', type=str, required=True, default='KINS_vqgan')
    parser.add_argument('--dataset', type=str, default="MOViD_A", help="select dataset")
    parser.add_argument('--data_type', type=str, default="image", help="select image or video model")
    parser.add_argument('--batch', type=int, default=1)

    args = parser.parse_args()

    if args.data_type == "image":
        from src.image_model import C2F_Seg
    elif args.data_type == "video":
        from src.video_model import C2F_Seg

    args.path = os.path.join(args.check_point_path, args.path)
    vq_model_path = os.path.join(args.check_point_path, args.vq_path)
    os.makedirs(args.path, exist_ok=True)

    config_path = os.path.join(args.path, 'c2f_seg_{}.yml'.format(args.dataset))
    if not os.path.exists(config_path):
        copyfile('./configs/c2f_seg_{}.yml'.format(args.dataset), config_path)
    
    config = Config(config_path)
    config.path = args.path
    config.batch_size = args.batch
    config.dataset = args.dataset

    log_file = 'log-{}.txt'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, this script supports only GPU execution.")

    config.device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    cv2.setNumThreads(0)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    test_dataset = load_dataset(config, args, "test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=1,
        drop_last=False
    )

    sample_iterator = test_dataset.create_iterator(config.sample_size)

    model = C2F_Seg(config, vq_model_path, mode='test', logger=logger)
    model.load(is_test=True, prefix=config.stage2_iteration)
    model.restore_from_stage1(prefix=config.stage1_iteration)

    model = model.to(config.device)

    iter = 0
    iou = 0
    iou_count = 0
    invisible_iou_ = 0
    occ_count = 0

    iou_post = 0
    iou_count_post = 0
    invisible_iou_post = 0
    occ_count_post = 0

    model.eval()
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        for items in test_loader:
            items = to_cuda(items, config.device)
            loss_eval = model.batch_predict_maskgit(items, iter, 'test', T=3)
            iter += 1
            iou += loss_eval['iou']
            iou_post += loss_eval['iou_post']
            iou_count += loss_eval['iou_count']
            invisible_iou_ += loss_eval['invisible_iou_']
            invisible_iou_post += loss_eval['invisible_iou_post']
            occ_count += loss_eval['occ_count']

            logger.info('iter {}: iou: {}, iou_post: {}, occ: {}, occ_post: {}'.format(
                iter-1,
                loss_eval['iou'].item(),
                loss_eval['iou_post'].item(),
                loss_eval['invisible_iou_'].item(),
                loss_eval['invisible_iou_post'].item(),
            ))
            torch.cuda.empty_cache()

    logger.info('meanIoU: {}'.format(iou.item() / iou_count.item()))
    logger.info('meanIoU post-process: {}'.format(iou_post.item() / iou_count.item()))
    logger.info('meanIoU invisible: {}'.format(invisible_iou_.item() / occ_count.item()))
    logger.info('meanIoU invisible post-process: {}'.format(invisible_iou_post.item() / occ_count.item()))
    logger.info('iou_count: {}'.format(iou_count))
    logger.info('occ_count: {}'.format(occ_count))
