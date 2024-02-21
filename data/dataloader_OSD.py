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

def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

class Fusion_OSD(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(Fusion_OSD, self).__init__()
        self.config = config
        self.mode = mode
        self.dataset_path = "/home/zhangjinyu/uoais_new/datasets/OSD-0.2-depth/"
        self.rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(self.dataset_path)))
        # can get occluded mask from annotation
        self.anno_paths = sorted(glob.glob("{}/annotation/*.png".format(self.dataset_path)))
        self.amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(self.dataset_path)))
        self.occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(self.dataset_path)))

        self.dtype = torch.float32
        self.enlarge_coef = 2
        self.patch_h = 256
        self.patch_w = 256
        self.device = "cpu"

        
    def __len__(self):
        return len(self.amodal_anno_paths)


    def __getitem__(self, index):
        return self.load_item(index)
        
    def load_item(self, index):
        # 获得anno并进行处理
        anno_file = self.amodal_anno_paths[index]
        amodal_anno = cv2.imread(anno_file)[...,0]
        img_name = anno_file.split('/')[-1].split('_')[0] + '.png'
        anno_id = int(anno_file.split('/')[-1].split('_')[1].strip('.png'))

        # 获得img
        img = cv2.imread(os.path.join("/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/image_color", img_name))
        # anno_id, img_path = self.label_info[index].split(",")
        height, width, _ = img.shape

        # ann = self.anns_dict_list[index]
        if "learn" in img_name:
            img_id = img_name[:-4].strip("learn")
        else:
            img_id = img_name[:-4].strip("test")
        img_id = int(img_id)
        # category_id = ann["category_id"]

        full_mask = amodal_anno>0
        # fm_no_crop = mask_utils.decode(full_mask)[...,np.newaxis]
        fm_no_crop = full_mask

        visible_mask = cv2.imread(os.path.join("/cpfs/2926428ee2463e44/user/zjy/data/OSD-0.2-depth/annotation", img_name))[...,0] == anno_id
        # vm_no_crop = mask_utils.decode(visible_mask)[...,np.newaxis]
        vm_no_crop = visible_mask

        if np.sum(vm_no_crop)==0:
            counts = np.array([0])
            print("DEBUG: all zeors: ",img_name)
            return dict()
        else:
            counts = np.array([1])
            y_min, x_min, w, h = get_bbox(full_mask)
            y_max, x_max = y_min + w, x_min + h
            y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max) 

            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_len = int((x_max - x_min) * self.enlarge_coef)
            y_len = int((y_max - y_min) * self.enlarge_coef)
            x_min = max(0, x_center - x_len // 2)
            x_max = min(height, x_center + x_len // 2)
            y_min = max(0, y_center - y_len // 2)
            y_max = min(width, y_center + y_len // 2)
            
            fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
            vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
            img_crop = img[x_min:x_max+1, y_min:y_max+1]

            h, w = vm_crop.shape[:2]
            m = transform.rescale(vm_crop, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
            vm_crop = m[np.newaxis, ...]

            img_ = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
            cur_h, cur_w = img_.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
            img_ = np.pad(img_, to_pad)[:self.patch_h, :self.patch_w, :3]
            img_crop = img_

            m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
            cur_h, cur_w = m.shape[:2]
            to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
            m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
            fm_crop = m[np.newaxis, ...]

            obj_position = np.array([x_min, x_max, y_min, y_max])
            vm_pad = np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)])
            vm_scale = np.array([self.patch_h/h, self.patch_w/w])

            # full_pad = ((0, max(375-height, 0)), (0, max(1242-width, 0)))
            # vm_no_crop = np.pad(vm_no_crop, full_pad)[:375, :1242]
            # fm_no_crop = np.pad(fm_no_crop, full_pad)[:375, :1242]
            vm_no_crop = vm_no_crop[np.newaxis, ...]
            fm_no_crop = fm_no_crop[np.newaxis, ...]

            # loss_mask = fm_no_crop-vm_no_crop
            # loss_mask[loss_mask==255]=0
            # loss_mask = 1-loss_mask.astype(bool)
            # data augmentation
            vm_crop_aug = self.data_augmentation(vm_crop[0])[np.newaxis, ...]
            
            counts = torch.from_numpy(counts).to(self.dtype).to(self.device)

            obj_position = torch.from_numpy(obj_position).to(self.dtype).to(self.device)
            vm_pad = torch.from_numpy(vm_pad).to(self.dtype).to(self.device)
            vm_scale = torch.from_numpy(vm_scale).to(self.dtype).to(self.device)

            fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)
            fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(self.dtype).to(self.device)
            vm_crop = torch.from_numpy(vm_crop).to(self.dtype).to(self.device)
            vm_crop_aug = torch.from_numpy(vm_crop_aug).to(self.dtype).to(self.device)
            img_crop = torch.from_numpy(img_crop).to(self.dtype).to(self.device)
            img = torch.from_numpy(img).to(self.dtype).to(self.device)
            vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(self.dtype).to(self.device)
            
            # loss_mask = torch.from_numpy(np.array(loss_mask)).to(self.dtype).to(self.device)
        
            img_id = torch.from_numpy(np.array(img_id)).to(self.dtype).to(self.device)
            anno_id = torch.from_numpy(np.array(anno_id)).to(self.dtype).to(self.device)
            # category_id = torch.from_numpy(np.array(category_id)).to(self.dtype).to(self.device)
            if self.mode=="train":
                meta = {
                    # "vm_no_crop": vm_no_crop,
                    "vm_crop": vm_crop_aug,
                    "vm_crop_gt": vm_crop,
                    # "fm_no_crop": fm_no_crop,
                    "fm_crop": fm_crop,
                    "img_crop": img_crop,
                    # "loss_mask": loss_mask,
                    "obj_position": obj_position,
                    "vm_pad": vm_pad,
                    "vm_scale": vm_scale,
                    "counts": counts,
                    "img_id": img_id,
                    # "anno_id": anno_id,
                    # "category_id": category_id,
                    # for vq
                    # "mask_crop": fm_crop
                    # "img_no_crop": img
                }
            elif self.mode=="test":
                meta = {
                    "vm_no_crop": vm_no_crop,
                    "vm_no_crop_gt": vm_no_crop,
                    "vm_crop": vm_crop,
                    "vm_crop_gt": vm_crop,
                    "fm_no_crop": fm_no_crop,
                    "fm_crop": fm_crop,
                    "img_crop": img_crop,
                    # "loss_mask": loss_mask,
                    "obj_position": obj_position,
                    "vm_pad": vm_pad,
                    "vm_scale": vm_scale,
                    "counts":counts,
                    "img_id": img_id,
                    # "anno_id": anno_id,
                    # "category_id": category_id,
                    # for vq
                    # "mask_crop": fm_crop
                    "img_no_crop": img,
                }
            return meta
        
    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                res[k] = default_collate(temp_)
            else:
                res[k] = None

        return res

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=self.collate_fn,
                shuffle=True
            )

            for item in sample_loader:
                yield item

    def polys_to_mask(self, polygons, height, width):
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask

    # def data_augmentation(self, mask):
    #     return mask
    
    def data_augmentation(self, mask):
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
    
    def make_json_dict(self, imgs, anns):
        imgs_dict = {}
        anns_dict = {}
        for ann in anns:
            image_id = ann["image_id"]
            if not image_id in anns_dict:
                anns_dict[image_id] = []
                anns_dict[image_id].append(ann)
            else:
                anns_dict[image_id].append(ann)
        
        for img in imgs:
            image_id = img['id']
            imgs_dict[image_id] = img['file_name']

        return imgs_dict, anns_dict



