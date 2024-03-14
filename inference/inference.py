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
    parser.add_argument('--dataset', type=str, default="MOViD_A", help="select dataset")
    parser.add_argument('--data_type', type=str, default="image", help="select image or video model")
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--model', type=str, default="original", help = "select model type")

    args = parser.parse_args()

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

    args.path = os.path.join(args.check_point_path, args.path)
    os.makedirs(args.path, exist_ok=True)

    config_path = os.path.join(args.path, 'c2f_seg_{}.yml'.format(args.dataset))
    
    config = Config(config_path)

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, this script supports only GPU execution.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    cv2.setNumThreads(0)

    from infdataset import inf_dataloader
    items = inf_dataloader(img_root_path="", depth_root_path="")

    model = C2F_Seg(config, mode='test', logger=logger)
    model.load(is_test=True, prefix=config.stage2_iteration)
    model = model.to(device)

    model.eval()

    items = to_cuda(items, device)
    pred_vm, pred_fm = model.inference(items)
    
    