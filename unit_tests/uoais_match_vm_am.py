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
import cvbase as cvb
from PIL import Image
from skimage import transform
from torch.utils.data.dataloader import default_collate
import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt
import glob
import imageio
from termcolor import colored
from tools import compute_PRF
import pickle

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    iou = intersection / union if union != 0 else 0
    return iou


if __name__ == "__main__":
    # load dataset
    dataset_path = "/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth"
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(dataset_path)))
    visible_anno_paths = sorted(glob.glob("{}/annotation/*.png".format(dataset_path)))
    with open('/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/pred_dict_list.pkl', 'rb') as f:
        uoais_visible_annos = pickle.load(f)   # 111
    amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(dataset_path)))
    occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(dataset_path)))
    for i, (rgb_path, depth_path, visible_instance) in enumerate(zip(tqdm(rgb_paths), depth_paths, uoais_visible_annos)):

        img_name = os.path.basename(rgb_path)[:-4] # 'learn0'
        filtered_visible_paths = list(filter(lambda p: img_name in p, visible_anno_paths))  # ['/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/annotation/learn0.png']
        filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths)) # ['/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/amodal_annotation/learn0_1.png', '/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/amodal_annotation/learn0_2.png']
        filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths)) # ['/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/occlusion_annotation/learn0_1.png']
    

        rgbs, depths, vms, annos = [], [], [], []
        uoais_vms, uoais_fms = visible_instance["pred_vm"], visible_instance["pred_fm"]
        for uoais_vm, uoais_fm in zip(uoais_vms, uoais_fms):
            max_iou, max_index = 0, -1
            for index, amodal_p in enumerate(filtered_amodal_paths):
                gt_fm = cv2.imread(amodal_p)[...,0] > 0
                cur_iou = compute_iou(gt_fm, uoais_fm)
                if cur_iou > max_iou:
                    max_iou, max_index = cur_iou, index
                    
            print(f"max iou is {max_iou}")