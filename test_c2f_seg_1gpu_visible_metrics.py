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
import os
import cv2
import random
import numpy as np
import cvbase as cvb
from PIL import Image
from skimage import transform
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt
import glob
import torch
import imageio
from termcolor import colored

def normalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm) 
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth

def unnormalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ unnormalize the input depth (0 ~ 255) and return depth image (mm)
    Args:
        depth([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.
    Returns:
        [np.float]: depth array [H, W] (mm) 
    """
    depth = np.float32(depth) / 255
    depth = depth * (max_val - min_val) + min_val
    return depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """ inpaint the input depth where the value is equal to zero

    Args:
        depth ([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        factor (int, optional): resize factor in depth inpainting. Defaults to 4.
        kernel_size (int, optional): kernel size in depth inpainting. Defaults to 5.

    Returns:
        [np.uint8]: inpainted depth array [H, W, 3] (0 ~ 255)
    """
    
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W//factor, H//factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth

def data_loader(rgb_path, depth_path):
    
    # load rgb and depth
    W, H = (640, 480)
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.resize(rgb_img, (W, H))
    depth_img = imageio.imread(depth_path)
    depth_img = normalize_depth(depth_img)
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)
    
    # # UOAIS-Net inference
    # if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
    #     uoais_input = depth_img
    # elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
    #     uoais_input = np.concatenate([rgb_img, depth_img], -1)   
    # else:
    #     uoais_input = rgb_img
    import ipdb; ipdb.set_trace()
    # laod GT (amodal masks)
    img_name = os.path.basename(rgb_path)[:-4] # 'learn0'
    annos = [] # [instance, IMG_H, IMG_W]
    filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths))
    filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths))

    for anno_path in filtered_amodal_paths:
        # get instance id
        inst_id = os.path.basename(anno_path)[:-4].split("_")[-1]
        inst_id = int(inst_id)
        # load mask image
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        # fill mask with instance id
        cnd = anno > 0
        anno_mask = np.zeros((H, W))
        anno_mask[cnd] = inst_id
        annos.append(anno_mask)            
    annos = np.stack(annos)
    num_inst_all_gt += len(filtered_amodal_paths)
    
    # 拿到vm
    # 按照vm数量复制对应的img和depth
    # ipdb> items["img_crop"].shape
    # torch.Size([32, 256, 256, 3])
    # depth形状相同
    
    
    return meta, annos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="../check_points")
    parser.add_argument('--dataset', type=str, default="MOViD_A", help="select dataset")
    parser.add_argument('--data_type', type=str, default="image", help="select image or video model")
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--model', type=str, default="original", help = "select model type")

    args = parser.parse_args()

    if args.data_type=="image":
        if args.model == "original":
            from src.image_model import C2F_Seg
        elif args.model == "rgbd_6channel":
            from src.image_model_depth_6channel import C2F_Seg
        elif args.model == "rgbd_fusion":
            from src.image_model_depth_fusion import C2F_Seg
        elif args.model == "depth_only":
            from src.image_model_depth import C2F_Seg
        elif args.model == "rgbd_linearfusion":
            from src.image_model_depth_linearfusion import C2F_Seg
    elif args.data_type == "video":
        from src.video_model import C2F_Seg

    args.path = os.path.join(args.check_point_path, args.path)
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

    # load dataset
    dataset_path = "/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth"
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(dataset_path)))
    visible_anno_paths = sorted(glob.glob("{}/annotation/*.png".format(dataset_path)))
    amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(dataset_path)))
    occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(dataset_path)))
    assert len(rgb_paths) == len(depth_paths)
    assert len(amodal_anno_paths) != 0
    assert len(occlusion_anno_paths) != 0
    print(colored("Evaluation on OSD dataset: {} rgbs, {} depths, {} amodal masks, {} occlusion masks".format(
                len(rgb_paths), len(depth_paths), len(amodal_anno_paths), len(occlusion_anno_paths)), "green"))

    metrics_all = [] # amodal mask evaluation
    num_inst_all_pred = 0 # number of all pred instances
    num_inst_all_gt = 0 # number of all GT instances
    num_inst_occ_pred = 0 # number of occluded prediction
    num_inst_occ_mat = 0 # number of occluded and matched
    num_inst_mat = 0 # number of matched instance
    
    mask_ious, occ_ious = 0, 0
    pre_occ, rec_occ, f_occ = 0, 0, 0
    pre_bou, rec_bou, f_bou = 0, 0, 0
    num_correct = 0 # for occlusion classification
    num_occ_over75 = 0 # number of instance IoU>0.75
    occ_over_75_rate = 0 # rate of instance IoU>0.75

    # model = C2F_Seg(config, mode='test', logger=logger)
    # model.load(is_test=True, prefix=config.stage2_iteration)
    # model = model.to(config.device)

    for i, (rgb_path, depth_path) in enumerate(zip(tqdm(rgb_paths), depth_paths)):
        # 根据rgb_path, depth_path处理meta

        meta = data_loader(rgb_path, depth_path)
        
        # pred_vm, pred_fm = model.calculate_metrics(meta)
        
        # pred_masks = instances.pred_masks.detach().cpu().numpy()
        # preds = [] # mask per each instance
        # for i, mask in enumerate(pred_masks):
        #     pred = np.zeros((H, W))
        #     pred[mask > False] = i+1
        #     preds.append(pred)
        #     num_inst_all_pred += 1