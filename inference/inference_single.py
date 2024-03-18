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
import re
from PIL import Image
import imageio

def add_mask(mask,img, color1, color_mask=np.array([0, 0, 255]),line_width=1):
    mask = mask.astype(np.bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    res = cv2.drawContours(img.copy(), contours, -1, color1, line_width)
    res[mask] = res[mask] * 0.7 + color_mask * 0.3
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--check_point_path', type=str, default="../check_points")
    parser.add_argument('--dataset', type=str, default="UOAIS", help="select dataset")
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
    
    model = C2F_Seg(config, mode='test')
    model.load(is_test=True, prefix=config.stage2_iteration)
    model = model.to(device)

    model.eval()        
    

    from infdataset import inf_dataloader
    
    # 指定路径
    directory_path = '/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/grounded_sam_output/one_head'
    output_path = '/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/lacnet_output/one_head'
    
    # 初始化两个列表，一个用于存储符合条件的完整文件名，另一个用于存储处理后的文件名部分
    visible_mask_files = []
    filename_prefixes = []

    # 编译一个正则表达式模式，用于匹配文件名结束于 "_visible_mask_0.jpg"、"_visible_mask_1.jpg" 等的文件名
    pattern = re.compile(r'_visible_mask_\d+\.jpg$')

    # 遍历指定路径下的所有文件
    for filename in os.listdir(directory_path):
        # 使用正则表达式检查文件名是否符合条件
        if pattern.search(filename):
            # 如果是，将文件名添加到列表中
            visible_mask_files.append(filename)
            
            # 处理文件名以提取 "banana_4" 等部分并保存
            # 使用正则表达式匹配到的字符串分割文件名并取第一部分
            prefix = re.split(r'_visible_mask_\d+\.jpg', filename)[0]
            filename_prefixes.append(prefix)
        
    img_path = os.path.join("/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/one_head/color", filename+".png")
    depth_path = os.path.join("/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/inference/one_head/depth", filename+".png")
    vm_path = os.path.join(directory_path, vm_file)
    
    items = inf_dataloader(img_path, depth_path, vm_path)

    items = to_cuda(items, device)
    _, pred_fm = model.inference(items)
    
    pred_fm_np = pred_fm.squeeze().cpu().numpy()
    
    # 转换数据类型为 uint8
    pred_fm_np_uint8 = (pred_fm_np * 255).astype(np.uint8)

    # 使用 Pillow 保存图像
    Image.fromarray(pred_fm_np_uint8).save(os.path.join(output_path, filename+"_lacnet_mask.png"))
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    color1 = [255, 0, 0]
    color2 = np.array(color1)+35
    masked_img = add_mask(pred_fm_np, img, color1, color2, 2) 
    imageio.imwrite(os.path.join(output_path, filename+"_lacnet_masked.png"), masked_img)  
    

    print(f"finished on {filename}")
    
