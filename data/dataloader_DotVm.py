# Kins dataset for Maskgit with image input center
class KinsImage_Dataset_transformer_img_center(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super(KinsImage_Dataset_transformer_img_center, self).__init__()
        self.config = config
        self.mode = mode

        root_path = "/home/ubuntu/data/Kins"
        flist = os.path.join(root_path, "{}_list.txt".format(mode))
        self.image_list = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        self.base_ann_path= os.path.join(root_path,"update_{}_2020.json".format(mode))
        self.base_img_path = os.path.join(root_path,"{}ing".format(mode),"image_2")
        annotations = cvb.load(self.base_ann_path)
        imgs_info = annotations['images']
        anns_info = annotations["annotations"]

        self.imgs_dict, self.anns_dict = self.make_json_dict(imgs_info, anns_info)

        self.data_list = list(self.anns_dict.keys())
        self.dtype = torch.float32
        self.enlarge_coef = 2
        self.patch_h = 256
        self.patch_w = 256
        self.device = "cpu"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return self.load_item(index)
    
    def mask_find_bboxs(self, mask):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
        stats = stats[stats[:,4].argsort()]
        return stats
    
    def generate_heatmap(self, mask, kernel, sigma):
        heatmap = cv2.GaussianBlur(mask, kernel, sigma)
        am = np.amax(heatmap)
        heatmap /= am / 1
        return heatmap
    
    def load_item(self, index):
        img_id, anno_id, category_id = self.image_list[index].split("_")
        img_id, anno_id, category_id = int(img_id), int(anno_id), int(category_id)

        img_name = self.imgs_dict[img_id]
        img_path = os.path.join(self.base_img_path, img_name)
        img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        
        ann = self.anns_dict[img_id][anno_id]
        fm_no_crop = self.polys_to_mask(ann["a_segm"], height, width)
        vm_no_crop = self.polys_to_mask(ann["i_segm"], height, width)

        counts = np.array([1])
        y_min, x_min, w, h = ann["i_bbox"]

        y_max, x_max = y_min + w, x_min + h
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        x_len = int((x_max - x_min) * self.enlarge_coef)
        y_len = int((y_max - y_min) * self.enlarge_coef)
        x_min = max(0, x_center - x_len // 2)
        x_max = min(height, x_center + x_len // 2)
        y_min = max(0, y_center - y_len // 2)
        y_max = min(width, y_center + y_len // 2)

        x_center_crop = x_center - x_min
        y_center_crop = y_center - y_min

        fm_crop = fm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
        vm_crop = vm_no_crop[x_min:x_max+1, y_min:y_max+1].astype(bool)
        img_crop = img[x_min:x_max+1, y_min:y_max+1]

        h, w = vm_crop.shape[:2]
        m = transform.rescale(vm_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]
        vm_crop = m[np.newaxis, ...]

        if np.sum(vm_no_crop)==0:
            center_crop = np.zeros_like(vm_crop)
            counts = np.array([0])
        else:
            center_crop = np.zeros_like(vm_crop[0])
            x_center_crop = int(x_center_crop*self.patch_h/h)
            y_center_crop = int(y_center_crop*self.patch_w/w)
            center_crop[x_center_crop: x_center_crop+1, y_center_crop: y_center_crop+1]=1
            center_crop = self.generate_heatmap(center_crop.astype(np.float), (35, 35), 9)
            center_crop = center_crop[np.newaxis, ...]

        m = transform.rescale(fm_crop, (self.patch_h/h, self.patch_w/w))
        cur_h, cur_w = m.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)))
        m = np.pad(m, to_pad)[:self.patch_h, :self.patch_w]    
        fm_crop = m[np.newaxis, ...]

        img_ = transform.rescale(img_crop, (self.patch_h/h, self.patch_w/w, 1))
        cur_h, cur_w = img_.shape[:2]
        to_pad = ((0, max(self.patch_h-cur_h, 0)), (0, max(self.patch_w-cur_w, 0)), (0, 0))
        img_ = np.pad(img_, to_pad)[:self.patch_h, :self.patch_w, :3]
        img_crop = img_

        obj_position = np.array([x_min, x_max, y_min, y_max])
        vm_pad = np.array([max(self.patch_h-cur_h, 0), max(self.patch_w-cur_w, 0)])
        vm_scale = np.array([self.patch_h/h, self.patch_w/w])

        vm_no_crop = vm_no_crop[np.newaxis, ...]
        fm_no_crop = fm_no_crop[np.newaxis, ...]

        loss_mask = fm_no_crop-vm_no_crop
        loss_mask[loss_mask==255]=0
        loss_mask = 1-loss_mask.astype(bool)
        
        counts = torch.from_numpy(counts).to(self.dtype).to(self.device)

        obj_position = torch.from_numpy(obj_position).to(self.dtype).to(self.device)
        vm_pad = torch.from_numpy(vm_pad).to(self.dtype).to(self.device)
        vm_scale = torch.from_numpy(vm_scale).to(self.dtype).to(self.device)

        fm_crop = torch.from_numpy(fm_crop).to(self.dtype).to(self.device)
        fm_no_crop = torch.from_numpy(np.array(fm_no_crop)).to(self.dtype).to(self.device)
        vm_crop = torch.from_numpy(vm_crop).to(self.dtype).to(self.device)
        vm_no_crop = torch.from_numpy(np.array(vm_no_crop)).to(self.dtype).to(self.device)
        img_crop = torch.from_numpy(np.array(img_crop)).to(self.dtype).to(self.device)
        center_crop = torch.from_numpy(np.array(center_crop)).to(self.dtype).to(self.device)

        loss_mask = torch.from_numpy(np.array(loss_mask)).to(self.dtype).to(self.device)

        img_id = torch.from_numpy(np.array(img_id)).to(self.dtype).to(self.device)
        anno_id = torch.from_numpy(np.array(anno_id)).to(self.dtype).to(self.device)
        category_id = torch.from_numpy(np.array(category_id)).to(self.dtype).to(self.device)
        
        if self.mode=="train":
            meta = {
                # "vm_no_crop": vm_no_crop,
                "vm_crop": vm_crop,
                # "fm_no_crop": fm_no_crop,
                "fm_crop": fm_crop,
                "img_crop": img_crop,
                "center_crop": center_crop,
                # "loss_mask": loss_mask,
                "obj_position": obj_position,
                "vm_pad": vm_pad,
                "vm_scale": vm_scale,
                "counts": counts,
                "img_id": img_id,
                "anno_id": anno_id,
                "category_id": category_id,
            }
        else:
            meta = {
                "vm_no_crop": vm_no_crop,
                "vm_crop": vm_crop,
                "fm_no_crop": fm_no_crop,
                "fm_crop": fm_crop,
                "img_crop": img_crop,
                "center_crop": center_crop,
                "loss_mask": loss_mask,
                "obj_position": obj_position,
                "vm_pad": vm_pad,
                "vm_scale": vm_scale,
                "counts":counts,
                "img_id": img_id,
                "anno_id": anno_id,
                "category_id": category_id,
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
                collate_fn=self.collate_fn
            )

            for item in sample_loader:
                yield item

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

    def polys_to_mask(self, polygons, height, width):
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask