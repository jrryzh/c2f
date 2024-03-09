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
from foreground_segmentation.model import Context_Guided_Network

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    iou = intersection / union if union != 0 else 0
    return iou

def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor

def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes
        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized

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

def data_loader(rgb_path, depth_path, visible_instance):    
    global num_inst_all_gt
    
    # 固定参数
    enlarge_coef = 2
    patch_h = 256
    patch_w = 256
    dtype = torch.float32
    device = "cuda"
    
    item_lst = []
    
    # 加载rgb和depth
    W, H = (640, 480)
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.resize(rgb_img, (W, H))
    depth_img = imageio.imread(depth_path)
    depth_img = normalize_depth(depth_img)
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)
    
    # 拿到对应visible amodal occlusion标注
    img_name = os.path.basename(rgb_path)[:-4] # 'learn0'
    filtered_visible_paths = list(filter(lambda p: img_name in p, visible_anno_paths))  # ['/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/annotation/learn0.png']
    filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths)) # ['/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/amodal_annotation/learn0_1.png', '/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/amodal_annotation/learn0_2.png']
    filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths)) # ['/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/occlusion_annotation/learn0_1.png']
    
    # 把fm和对应vm进行匹配
    # 做unit test看匹配情况如何
    rgbs, depths, vms, fms = [], [], [], []  # 分别用于存储RGB图像、深度图、可视化掩模和注释
    uoais_vms, uoais_fms = visible_instance["pred_vm"], visible_instance["pred_fm"]  # 从预测的实例中获取可视化掩模和前景掩模

    # 遍历每个可视化掩模和前景掩模
    for uoais_vm, uoais_fm in zip(uoais_vms, uoais_fms):
        max_iou, max_index, max_fm = 0, -1, None  # 初始化最大IOU和对应的索引和注释
        for index, amodal_p in enumerate(filtered_amodal_paths):     
            # 从文件路径中解析实例ID
            inst_id = os.path.basename(amodal_p)[:-4].split("_")[-1]
            inst_id = int(inst_id)
            # 加载并调整掩模图像尺寸
            fm = imageio.imread(amodal_p)
            fm = cv2.resize(fm, (W, H), interpolation=cv2.INTER_NEAREST)
            # 将掩模填充为实例ID
            cnd = fm > 0
            fm_mask = np.zeros((H, W))
            fm_mask[cnd] = inst_id
                            
            # 计算当前IOU
            cur_iou = compute_iou(fm_mask, uoais_fm)
            if cur_iou > max_iou:
                max_iou, max_index, max_fm = cur_iou, index, fm_mask
                    
        if max_iou > 0.5:  # 如果IOU大于0.5，则认为是有效匹配
            vms.append(uoais_vm)
            fms.append(max_fm)
            rgbs.append(rgb_img)
            depths.append(depth_img)
    fms = np.stack(fms)  # 将注释列表转换为numpy数组
    
    # 计算annos
    annos = []
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
    
    # 迭代img depth vm fm 进行inference
    for img, depth, vm, fm in zip(rgbs, depths, vms, fms):
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

        vm_no_crop = vm_no_crop[np.newaxis, ...]
        fm_no_crop = fm_no_crop[np.newaxis, ...]

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
    with open('/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/pred_dict_list.pkl', 'rb') as f:
        uoais_visible_annos = pickle.load(f)   # 111
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

    # 加载C2F模型
    model = C2F_Seg(config, mode='test', logger=logger)
    model.load(is_test=True, prefix=config.stage2_iteration)
    model = model.to(config.device)
    
    # 加载foreground model
    checkpoint = torch.load("/cpfs/2926428ee2463e44/user/zjy/code_repo/c2f-seg/foreground_segmentation/rgbd_fg.pth")
    fg_model = Context_Guided_Network(classes=2, in_channel=4)
    fg_model.load_state_dict(checkpoint['model'])
    fg_model.cuda()
    fg_model.eval()

    for i, (rgb_path, depth_path, visible_instance) in enumerate(zip(tqdm(rgb_paths), depth_paths, uoais_visible_annos)):
        # 根据rgb_path, depth_path处理meta
        meta_lst, annos, filtered_occlusion_paths, img_name, vms = data_loader(rgb_path, depth_path, visible_instance)
        
        pred_vm_lst, pred_fm_lst = model.calculate_metrics(meta_lst)
        
        pred_vm_lst = [pv.detach().cpu().numpy() for pv in pred_vm_lst]
        pred_fm_lst = [pf.detach().cpu().numpy() for pf in pred_fm_lst]
        
        # load rgb and depth
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
        fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
        fg_depth_input = cv2.resize(depth_img, (320, 240)) 
        fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
        fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
        fg_output = fg_model(fg_input.cuda())
        fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
        fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
        fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

        # 用foreground检测筛选
        pred_vms, pred_fms = [], []
        for i, (pred_fm, pred_vm) in enumerate(zip(pred_fm_lst, pred_vm_lst)):
            overlap = np.sum(np.bitwise_and(pred_fm, fg_output)) / np.sum(pred_fm)
            if overlap >= 0.5: # filiter outliers
                pred = np.zeros((H, W))
                pred[pred_fm > False] = i+1
                pred_fms.append(pred.astype("int64"))
                pred_vms.append(pred_vm)
                num_inst_all_pred += 1
        if len(pred_fms) > 0:
            pred_fms = np.stack(pred_fms)
            pred_vms = np.stack(pred_vms)
        else:
            pred_fms = np.array(pred_fms)
            pred_vms = np.array(pred_vms)
        
        metrics, assignments = compute_PRF.multilabel_amodal_metrics(pred_fms, annos, return_assign=True)
        metrics_all.append(metrics)
        
        num_inst_mat += len(assignments)

        amodals = pred_fm_lst
        visibles = pred_vm_lst
        # TODO: SHit code
        preds = pred_fms
        
        # count occluded area of predictions when classified
        all_occ_pred, all_bou_pred = 0, 0
        num_inst_occ_prd_img = 0
        for pred in preds:
            idx = int(pred.max())-1
            try:
                amodal = amodals[idx]
                visible = visibles[idx]
                occ = np.bitwise_xor(amodal, visible)
            except:
                import ipdb; ipdb.set_trace()
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