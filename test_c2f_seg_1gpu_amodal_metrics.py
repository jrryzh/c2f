"""
用于计算uoais文章中的amodal metrics
存在两个问题 1. h, w存在冗余 2. num_inst_all_gt 设置成global 3. dataload 方式可以改进
运行不存在问题 后续需要改进
"""

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

def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

def data_augmentation(mask):
    np.float = float    
    mask = mask.astype(np.float)
    rdv = random.random()
    n_repeat = random.randint(1, 4)
    if rdv <= 0.2:
        mask = cv2.GaussianBlur(mask, (35,35), 11)
    elif rdv > 0.2 and rdv <0.9:
        rdv_1 = random.random()
        rdv_2 = random.random()
        for i in range(n_repeat):
            w = random.randint(5, 13)
            h = random.randint(5, 13)
            kernel = np.ones((w, h), dtype=np.uint8)
            if rdv_1 <= 0.6:
                mask = cv2.dilate(mask, kernel, 1)
            elif rdv_1 > 0.6 and rdv_1 <= 1.0:
                mask = cv2.erode(mask, kernel, 1)
            if rdv_2 <= 0.2:
                mask = cv2.GaussianBlur(mask, (35,35), 11)
    else:
        mask = mask
    return (mask>0.5)

def data_loader(rgb_path, depth_path):
    global num_inst_all_gt
    
    # 固定参数
    enlarge_coef = 2
    patch_h = 256
    patch_w = 256
    dtype = torch.float32
    device = "cuda"
    
    item_lst = []
    
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

    # load GT (amodal masks)
    img_name = os.path.basename(rgb_path)[:-4] # 'learn0'
    annos = [] # [instance, IMG_H, IMG_W]
    filtered_visible_paths = list(filter(lambda p: img_name in p, visible_anno_paths))
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
    visible_mask_img = cv2.imread(filtered_visible_paths[0])[...,0]
    vms = []
    for anno_id in range(len(filtered_amodal_paths)):
        vms.append(visible_mask_img == anno_id+1)
    # vms = np.stack(vms)
    
    assert len(vms) == len(filtered_amodal_paths)
    rgbs, depths = [], []
    for _ in range(len(filtered_amodal_paths)):
        rgbs.append(rgb_img)
        depths.append(depth_img)
    
    # 按照vm数量复制对应的img和depth
    # ipdb> items["img_crop"].shape
    # torch.Size([32, 256, 256, 3])
    # depth形状相同
    
    for img, depth, vm, fm in zip(rgbs, depths, vms, annos):
        # TODO: shit code 
        counts = np.array([1])
        height, width, _ = img.shape
        vm_no_crop = vm
        fm_no_crop = fm
        full_mask = fm > 0

        y_min, x_min, w, h = get_bbox(full_mask)
        y_max, x_max = y_min + w, x_min + h
        y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max) 

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        x_len = int((x_max - x_min) * enlarge_coef)
        y_len = int((y_max - y_min) * enlarge_coef)
        x_min = max(0, x_center - x_len // 2)
        x_max = min(height, x_center + x_len // 2)
        y_min = max(0, y_center - y_len // 2)
        y_max = min(width, y_center + y_len // 2)
        
        fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
        vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
        img_crop = img[x_min:x_max+1, y_min:y_max+1]
        depth_crop = depth[x_min:x_max+1, y_min:y_max+1]
        h, w = vm_crop.shape[:2]
        m = transform.rescale(vm_crop, (patch_h/h, patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(patch_h-cur_h, 0)), (0, max(patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:patch_h, :patch_w]
        vm_crop = m[np.newaxis, ...]

        img_ = transform.rescale(img_crop, (patch_h/h, patch_w/w, 1))
        cur_h, cur_w = img_.shape[:2]
        to_pad = ((0, max(patch_h-cur_h, 0)), (0, max(patch_w-cur_w, 0)), (0, 0))
        img_ = np.pad(img_, to_pad)[:patch_h, :patch_w, :3]
        img_crop = img_
        # 修改：添加depth
        depth_ = transform.rescale(depth_crop, (patch_h/h, patch_w/w, 1))
        cur_h, cur_w = depth_.shape[:2]
        to_pad = ((0, max(patch_h-cur_h, 0)), (0, max(patch_w-cur_w, 0)), (0, 0))
        depth_ = np.pad(depth_, to_pad)[:patch_h, :patch_w, :3]
        depth_crop = depth_
        m = transform.rescale(fm_crop, (patch_h/h, patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(patch_h-cur_h, 0)), (0, max(patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:patch_h, :patch_w]    
        fm_crop = m[np.newaxis, ...]

        obj_position = np.array([x_min, x_max, y_min, y_max])
        vm_pad = np.array([max(patch_h-cur_h, 0), max(patch_w-cur_w, 0)])
        vm_scale = np.array([patch_h/h, patch_w/w])

        # full_pad = ((0, max(375-height, 0)), (0, max(1242-width, 0)))
        # vm_no_crop = np.pad(vm_no_crop, full_pad)[:375, :1242]
        # fm_no_crop = np.pad(fm_no_crop, full_pad)[:375, :1242]
        vm_no_crop = vm_no_crop[np.newaxis, ...]
        fm_no_crop = fm_no_crop[np.newaxis, ...]

        # loss_mask = fm_no_crop-vm_no_crop
        # loss_mask[loss_mask==255]=0
        # loss_mask = 1-loss_mask.astype(bool)
        # data augmentation
        vm_crop_aug = data_augmentation(vm_crop[0])[np.newaxis, ...]
        
        counts = torch.from_numpy(counts).to(dtype).to(device)

        obj_position = torch.from_numpy(obj_position).to(dtype).to(device)
        vm_pad = torch.from_numpy(vm_pad).to(dtype).to(device)
        vm_scale = torch.from_numpy(vm_scale).to(dtype).to(device)

        fm_crop = torch.from_numpy(fm_crop).to(dtype).to(device)
        fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(dtype).to(device)
        vm_crop = torch.from_numpy(vm_crop).to(dtype).to(device)
        vm_crop_aug = torch.from_numpy(vm_crop_aug).to(dtype).to(device)
        img_crop = torch.from_numpy(img_crop).to(dtype).to(device)
        img = torch.from_numpy(img).to(dtype).to(device)
        # 修改： 添加depth
        depth_crop = torch.from_numpy(depth_crop).to(dtype).to(device)
        depth = torch.from_numpy(depth).to(dtype).to(device)
        vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(dtype).to(device)
        
        # category_id = torch.from_numpy(np.array(category_id)).to(dtype).to(device)
        meta = {
            "vm_no_crop": vm_no_crop.unsqueeze(0),
            "vm_no_crop_gt": vm_no_crop.unsqueeze(0),
            "vm_crop": vm_crop.unsqueeze(0),
            "vm_crop_gt": vm_crop.unsqueeze(0),
            "fm_no_crop": fm_no_crop.unsqueeze(0),
            "fm_crop": fm_crop.unsqueeze(0),
            "img_crop": img_crop.unsqueeze(0),
            # "loss_mask": loss_mask,
            "depth_crop": depth_crop.unsqueeze(0),    # 修改： 添加depth
            "obj_position": obj_position.unsqueeze(0),
            "vm_pad": vm_pad.unsqueeze(0),
            "vm_scale": vm_scale.unsqueeze(0),
            "counts":counts.unsqueeze(0),
            # "anno_id": anno_id,
            # "category_id": category_id,
            # for vq
            # "mask_crop": fm_crop
            "img_no_crop": img.unsqueeze(0),
        }
        item_lst.append(meta)
    return item_lst, annos, filtered_occlusion_paths, img_name, vms
    
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
    # TODO: shit code
    W, H = 640, 480

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

    model = C2F_Seg(config, mode='test', logger=logger)
    model.load(is_test=True, prefix=config.stage2_iteration)
    model = model.to(config.device)

    for i, (rgb_path, depth_path) in enumerate(zip(tqdm(rgb_paths), depth_paths)):
        # 根据rgb_path, depth_path处理meta

        meta_lst, annos, filtered_occlusion_paths, img_name, vms = data_loader(rgb_path, depth_path)
        
        pred_vm_lst, pred_fm_lst = model.calculate_metrics(meta_lst)
        
        pred_vm_lst = [pv.detach().cpu().numpy() for pv in pred_vm_lst]
        pred_fm_lst = [pf.detach().cpu().numpy() for pf in pred_fm_lst]
        preds = []

        for i, mask in enumerate(pred_fm_lst):
            pred = np.zeros_like(mask)
            pred[mask > 0] = i+1
            preds.append(pred)
            num_inst_all_pred += 1
        preds = np.stack(preds)
        
        assert mask.shape == (H, W)

        metrics, assignments = compute_PRF.multilabel_amodal_metrics(preds, annos, return_assign=True)
        metrics_all.append(metrics)
        
        num_inst_mat += len(assignments)

        amodals = pred_fm_lst
        # visibles = vms 
        visibles = pred_vm_lst 
        
        # count occluded area of predictions when classified
        all_occ_pred, all_bou_pred = 0, 0
        num_inst_occ_prd_img = 0
        for pred in preds:
            idx = int(pred.max())-1
            amodal = amodals[idx]
            visible = visibles[idx]
            import ipdb; ipdb.set_trace()
            occ = np.bitwise_xor(amodal, visible)
            try:
                cls = instances.pred_occlusions[idx].item()
            except:
                # if area over 5% of amodal mask is not visible
                cls = 1 if np.int64(np.count_nonzero(occ)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0                
            if not cls: continue
            num_inst_occ_pred += 1
            num_inst_occ_prd_img += 1
            all_occ_pred += np.int64(np.count_nonzero(occ))
            all_bou_pred += np.sum(compute_PRF.seg2bmap(occ))
        # count occluded area of ground truth
        all_occ_gt, all_bou_gt = 0, 0
        occ_paths = filtered_occlusion_paths

        for occ_path in occ_paths:
            occ = imageio.imread(occ_path)
            occ = cv2.resize(occ, (W, H), interpolation=cv2.INTER_NEAREST)
            occ = occ[:,:] > 0
            all_occ_gt += np.int64(np.count_nonzero(occ))
            all_bou_gt += np.sum(compute_PRF.seg2bmap(occ))
            
            idx = int(os.path.basename(occ_path)[:-4].split("_")[1]) - 1
        # count area with matched instances
        # assign: [[gt_id, pred_id], ... ]
        assign_amodal_pred, assign_visible_pred, assign_amodal_gt = 0, 0, 0
        assign_occ_pred, assign_occ_gt = 0, 0
        assign_amodal_overlap, assign_occ_overlap = 0, 0
        occ_bou_pre, occ_bou_rec = 0, 0
        num_occ_over75_img = 0
       
        for gt_id, pred_id in assignments:              
            ###############
            # AMODAL MASK #
            ###############
            # count area of masks predictions
            amodal = amodals[int(pred_id)-1]
            visible = visibles[int(pred_id)-1]
            assign_amodal_pred += np.count_nonzero(amodal)
            assign_visible_pred += np.count_nonzero(visible)
            # count area of mask GT [ annos: amodal GT ]
            anno = annos[np.where(annos == gt_id)[0][0]] > 0
            assign_amodal_gt += np.count_nonzero(anno)
            # count overlap area of mask btw. pred & GT
            amodal_overlap = np.logical_and(amodal, anno)
            assign_amodal_overlap += np.count_nonzero(amodal_overlap)
                        
            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion prediction
            occ_pred = np.bitwise_xor(amodal, visible)
            assign_occ_pred += np.count_nonzero(occ_pred)

            ############################
            # OCCLUSION CLASSIFICATION #
            ############################
            # count occlusion classification corrects
            cls_gt = os.path.isfile("{}/occlusion_annotation/{}_{}.png".format(
                                    dataset_path, img_name, int(gt_id)))
            try:
                cls_pred = instances.pred_occlusions[int(pred_id)-1].item()
            except:
                cls_pred = 1 if np.int64(np.count_nonzero(occ_pred)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0
            num_correct += cls_pred==cls_gt
            if cls_pred==cls_gt and cls_pred == 1:
                num_inst_occ_mat += cls_pred         

            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion GT
            occ_path = "{}/occlusion_annotation/{}_{}.png".format(
                                    dataset_path, img_name, int(gt_id))
            if not os.path.isfile(occ_path) or not cls_pred: continue
            occ_gt = imageio.imread(occ_path)
            occ_gt = cv2.resize(occ_gt, (W, H), interpolation=cv2.INTER_NEAREST)
            occ_gt = occ_gt[:,:] > 0
            assign_occ_gt += np.count_nonzero(occ_gt)
            # count overlap area of occlusion btw. pred & GT
            occ_overlap = np.logical_and(occ_pred, occ_gt)
            assign_occ_overlap += np.count_nonzero(occ_overlap)
            
            ############################
            # Over75 of OCCLUSION MASK #
            ############################
            pre = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_pred)
            rec = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_gt)
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            if f > 0.75: 
                num_occ_over75_img += 1
                num_occ_over75 += 1
                        
            ###########################
            # OCCLUSION MASK BOUNDARY #
            ###########################
            pre, rec = compute_PRF.boundary_overlap_occ(gt_id, pred_id, occ_pred, occ_gt)
            occ_bou_pre += pre
            occ_bou_rec += rec
        
        #####################################
        # COMPUTE METRICS in a single image #
        #####################################
        # mIoU of amodal mask (only matched instance)
        if assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap > 0:
            iou = assign_amodal_overlap / (assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap)
        else: iou = 0
        mask_ious += iou
        # mIoU of occlusion mask (only matched instance)
        if assign_occ_pred+assign_occ_gt-assign_occ_overlap > 0:
            iou = assign_occ_overlap / (assign_occ_pred+assign_occ_gt-assign_occ_overlap)
        else: iou = 0
        occ_ious +=iou
        
        # number of occluded instances in one image
        num_pred = num_inst_occ_prd_img
        num_gt = len(filtered_occlusion_paths)
        if num_pred == 0 and num_gt > 0:
            pre_occ += 1
            pre_bou += 1
        elif num_pred > 0 and num_gt == 0:
            rec_occ += 1
            rec_bou += 1
        elif num_pred == 0 and num_gt == 0:
            pre_occ += 1
            rec_occ += 1
            f_occ += 1
            pre_bou += 1
            rec_bou += 1
            f_bou += 1
            occ_over_75_rate += 1
        else:
            assert (num_pred > 0) and (num_gt > 0)
            # P, R, F of occlusion mask (all instance)
            pre = assign_occ_overlap / all_occ_pred if all_occ_pred > 0 else 0
            rec = assign_occ_overlap / all_occ_gt if all_occ_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_occ += pre
            rec_occ += rec
            f_occ += f
            # P, R, F of occlusion boundary (all instance)
            pre = occ_bou_pre / all_bou_pred if all_bou_pred > 0 else 0
            rec = occ_bou_rec / all_bou_gt if all_bou_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_bou += pre
            rec_bou += rec
            f_bou += f
            occ_over_75_rate += num_occ_over75_img / num_gt

    ###############################################
    # get average metirc values among test images #
    ###############################################
    num = len(metrics_all)
    mask_ious /= num 
    occ_ious /= num 
    pre_occ /= num 
    rec_occ /= num 
    f_occ /= num 
    pre_bou /= num 
    rec_bou /= num 
    f_bou /= num 
    occ_over_75_rate /= num
    
    occ_cls_acc = num_correct / num_inst_mat * 100 if num_inst_mat > 0 else 0
    occ_cls_pre = num_correct / num_inst_all_pred * 100 if num_inst_all_pred > 0 else 0
    occ_cls_rec = num_correct / num_inst_all_gt * 100 if num_inst_all_gt > 0 else 0
    occ_cls_f = (2*occ_cls_pre*occ_cls_rec) / (occ_cls_pre+occ_cls_rec) if occ_cls_pre+occ_cls_rec > 0 else 0

    # sum the values with same keys
    result = {}
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Amodal Metrics on OSD (for {} instances)".format(num_inst_all_gt), "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, mask_ious
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
    print(colored("Occlusion Metrics on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        pre_occ*100, rec_occ*100, f_occ*100, 
        pre_bou*100, rec_bou*100, f_bou*100,
        occ_over_75_rate*100, occ_ious        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    print(colored("Occlusion Classification on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("  P   R   F   ACC")
    print("{:.1f} {:.1f} {:.1f} {:.1f}".format(
        occ_cls_pre, occ_cls_rec, occ_cls_f, occ_cls_acc        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')