import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms

from taming_src.taming_models import VQModel
from src.image_component import MaskedTransformer, Resnet_Encoder, RGBD_Resnet_Encoder, Refine_Module
from src.loss import VGG19, PerceptualLoss
from utils.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from utils.utils import torch_show_all_params, torch_init_model
from utils.utils import Config
from utils.evaluation import evaluation_image
from utils.loss import CrossEntropyLoss


class C2F_Seg(nn.Module):
    def __init__(self, config, mode, logger=None, save_eval_dict={}):
        super(C2F_Seg, self).__init__()
        self.config = config
        self.iteration = 0
        self.sample_iter = 0
        self.name = config.model_type

        self.root_path = config.path
        self.transformer_path = os.path.join(config.path, self.name)

        self.mode = mode
        self.save_eval_dict = save_eval_dict

        self.eps = 1e-6
        self.train_sample_iters = config.train_sample_iters
        
        self.encoder = RGBD_Resnet_Encoder().to(config.device)
        self.refine_module = Refine_Module().to(config.device)

        self.refine_criterion = nn.BCELoss()
        self.criterion = CrossEntropyLoss(num_classes=config.vocab_size+1, device=config.device)

        if config.train_with_dec:
            if not config.gumbel_softmax:
                self.temperature = nn.Parameter(torch.tensor([config.tp], dtype=torch.float32),
                                                requires_grad=True).to(config.device)
            if config.use_vgg:
                vgg = VGG19(pretrained=True, vgg_norm=config.vgg_norm).to(config.device)
                vgg.eval()
                reduction = 'mean' if config.balanced_loss is False else 'none'
                self.perceptual_loss = PerceptualLoss(vgg, weights=config.vgg_weights,
                                                      reduction=reduction).to(config.device)
        else:
            self.perceptual_loss = None

        # loss
        param_optimizer_encoder = self.encoder.named_parameters()
        param_optimizer_refine= self.refine_module.named_parameters()
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer_encoder], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer_refine], 'weight_decay': config.weight_decay},
        ]

        self.opt = AdamW(params=optimizer_parameters,
                         lr=float(config.lr), betas=(config.beta1, config.beta2))
        self.sche = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_iters,
                                                    num_training_steps=config.max_iters)

        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0
            
        self.gamma = self.gamma_func(mode=config.gamma_mode)
        self.mask_token_idx = config.vocab_size
        self.choice_temperature = 4.5
        self.Image_W = config.Image_W
        self.Image_H = config.Image_H
        self.patch_W = config.patch_W
        self.patch_H = config.patch_H

    @torch.no_grad()
    def encode_to_z(self, x, mask=None):
        if len(x.size())==5:
            x = x[0]
        quant_z, _, info = self.g_model.encode(x.float(), mask)  # [B,D,H,W]
        indices = info[2].view(quant_z.shape[0], -1)  # [B, L]
        return quant_z, indices
    
    def get_attn_map(self, feature, guidance):
        guidance = F.interpolate(guidance, scale_factor=(1/16))
        b,c,h,w = guidance.shape
        q = torch.flatten(guidance, start_dim=2)
        v = torch.flatten(feature, start_dim=2)

        k = v * q
        k = k.sum(dim=-1, keepdim=True) / (q.sum(dim=-1, keepdim=True) + 1e-6)
        attn = (k.transpose(-2, -1) @  v) / 1
        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(b, c, h, w)
        return attn

    def get_losses(self, meta):
        self.iteration += 1
        rgbd_crop = torch.cat((meta["img_crop"], meta["depth_crop"]), dim=-1)
        rgbd_feat = self.encoder(rgbd_crop.permute((0,3,1,2)).to(torch.float32))
        
        # 修改： 将原来的transformer预测的coarse mask改为vm_crop_gt
        pred_fm_crop = meta["vm_crop_gt"]
        pred_vm_crop, pred_fm_crop = self.refine_module(rgbd_feat, pred_fm_crop.detach())
        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
        pred_vm_crop = torch.sigmoid(pred_vm_crop)
        loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt'])
        # pred_vm_crop = (pred_vm_crop>=0.5).to(torch.float32)

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
        pred_fm_crop = torch.sigmoid(pred_fm_crop)
        loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'])
        # pred_fm_crop = (pred_fm_crop>=0.5).to(torch.float32)
        logs = [
            ("loss_vm", loss_vm.item()),
            ("loss_fm", loss_fm.item()),
        ]
        return loss_vm+loss_fm, logs
    
    def align_raw_size(self, full_mask, obj_position, vm_pad, meta):
        vm_np_crop = meta["vm_no_crop"].squeeze()
        H, W = vm_np_crop.shape[-2], vm_np_crop.shape[-1]
        bz, seq_len = full_mask.shape[:2]
        new_full_mask = torch.zeros((bz, seq_len, H, W)).to(torch.float32).cuda()
        if len(vm_pad.shape)==3:
            vm_pad = vm_pad[0]
            obj_position = obj_position[0]
        for b in range(bz):
            paddings = vm_pad[b]
            position = obj_position[b]
            new_fm = full_mask[
                b, :,
                :-int(paddings[0]) if int(paddings[0]) !=0 else None,
                :-int(paddings[1]) if int(paddings[1]) !=0 else None
            ]
            vx_min = int(position[0])
            vx_max = min(H, int(position[1])+1)
            vy_min = int(position[2])
            vy_max = min(W, int(position[3])+1)
            resize = transforms.Resize([vx_max-vx_min, vy_max-vy_min])
            try:
                new_fm = resize(new_fm)
                new_full_mask[b, :, vx_min:vx_max, vy_min:vy_max] = new_fm[0]
            except:
                new_fm = new_fm
        return new_full_mask

    def loss_and_evaluation(self, pred_fm, meta, iter, mode, pred_vm=None):
        loss_eval = {}
        pred_fm = pred_fm.squeeze()
        counts = meta["counts"].reshape(-1).to(pred_fm.device)
        fm_no_crop = meta["fm_no_crop"].squeeze()
        vm_no_crop = meta["vm_no_crop"].squeeze()
        pred_vm = pred_vm.squeeze()
        # post-process
        pred_fm = (pred_fm > 0.5).to(torch.int64)
        pred_vm = (pred_vm > 0.5).to(torch.int64)
        
        iou, invisible_iou_, iou_count = evaluation_image((pred_fm > 0.5).to(torch.int64), fm_no_crop, counts, meta, self.save_eval_dict)
        loss_eval["iou"] = iou
        loss_eval["invisible_iou_"] = invisible_iou_
        loss_eval["occ_count"] = iou_count
        loss_eval["iou_count"] = torch.Tensor([pred_fm.shape[0]]).cuda()
        pred_fm_post = pred_fm + vm_no_crop
        
        pred_fm_post = (pred_fm_post>0.5).to(torch.int64)
        iou_post, invisible_iou_post, iou_count_post = evaluation_image(pred_fm_post, fm_no_crop, counts, meta, self.save_eval_dict)
        loss_eval["iou_post"] = iou_post
        loss_eval["invisible_iou_post"] = invisible_iou_post
        return loss_eval

    def backward(self, loss=None):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sche.step()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def batch_predict_maskgit(self, meta, iter, mode, T=3, start_iter=0):
        '''
        :param x:[B,3,H,W] image
        :param c:[b,X,H,W] condition
        :param mask: [1,1,H,W] mask
        '''
        self.sample_iter += 1

        rgbd_crop = torch.cat((meta["img_crop"], meta["depth_crop"]), dim=-1)
        rgbd_feat = self.encoder(rgbd_crop.permute((0,3,1,2)).to(torch.float32))
        
        # 修改： 将原来的transformer预测的coarse mask改为vm_crop_gt
        pred_fm_crop_old = meta["vm_crop_gt"]
        pred_vm_crop, pred_fm_crop = self.refine_module(rgbd_feat, pred_fm_crop_old)

        pred_vm_crop = F.interpolate(pred_vm_crop, size=(256, 256), mode="nearest")
        pred_vm_crop = torch.sigmoid(pred_vm_crop)
        loss_vm = self.refine_criterion(pred_vm_crop, meta['vm_crop_gt'])
        # pred_vm_crop = (pred_vm_crop>=0.5).to(torch.float32)

        pred_fm_crop = F.interpolate(pred_fm_crop, size=(256, 256), mode="nearest")
        pred_fm_crop = torch.sigmoid(pred_fm_crop)
        loss_fm = self.refine_criterion(pred_fm_crop, meta['fm_crop'])
        # pred_fm_crop = (pred_fm_crop>=0.5).to(torch.float32)

        pred_vm = self.align_raw_size(pred_vm_crop, meta['obj_position'], meta["vm_pad"], meta)
        pred_fm = self.align_raw_size(pred_fm_crop, meta['obj_position'], meta["vm_pad"], meta)
        
        # visualization
        # self.visualize(pred_vm, pred_fm, meta, mode, iter)

        loss_eval = self.loss_and_evaluation(pred_fm, meta, iter, mode, pred_vm=pred_vm)
        loss_eval["loss_fm"] = loss_fm
        loss_eval["loss_vm"] = loss_vm
        return loss_eval


    def visualize(self, pred_vm, pred_fm, meta, mode, iteration):
        # import ipdb; ipdb.set_trace()
        pred_fm = pred_fm.squeeze()
        pred_vm = pred_vm.squeeze()
        gt_vm = meta["vm_no_crop"].squeeze()
        gt_fm = meta["fm_no_crop"].squeeze()
        to_plot = torch.cat((pred_vm, pred_fm, gt_vm, gt_fm)).cpu().numpy()
        save_dir = os.path.join(self.root_path, '{}_samples'.format(mode))
        image_id, anno_id= meta["img_id"], meta["anno_id"]
        plt.imsave("{}/{}_{}_{}.png".format(save_dir, iteration, int(image_id.item()), int(anno_id.item())), to_plot)
    
    # def visualize_crop(self, pred_vm, pred_fm, meta, mode, count, pred_fm_crop_old):
    #     pred_fm = pred_fm.squeeze()
    #     pred_vm = pred_vm.squeeze()
    #     pred_fm_crop_old = pred_fm_crop_old.squeeze()
    #     gt_vm = meta["vm_crop"].squeeze()
    #     gt_fm = meta["fm_crop"].squeeze()
    #     to_plot = torch.cat((pred_vm, gt_vm, pred_fm_crop_old, pred_fm, gt_fm)).cpu().numpy()
    #     save_dir = os.path.join(self.root_path, '{}_samples'.format(mode))
    #     image_id, anno_id= meta["img_id"], meta["anno_id"]
    #     plt.imsave("{}/{}_{}_{}_{}.png".format(save_dir, count, int(image_id.item()), int(anno_id.item()), "crop"), to_plot)

    def create_inputs_tokens_normal(self, num, device):
        self.num_latent_size = self.config['resolution'] // self.config['patch_size']
        blank_tokens = torch.ones((num, self.num_latent_size ** 2), device=device)
        masked_tokens = self.mask_token_idx * blank_tokens

        return masked_tokens.to(torch.int64)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        elif mode == "log":
            return lambda r, total_unknown: - np.log2(r) / np.log2(total_unknown)
        else:
            raise NotImplementedError

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1) # from small to large
        # Obtains cut off threshold given the mask lengths.
        # cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        cut_off = sorted_confidence.gather(dim=-1, index=mask_len.to(torch.long))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking
    
    def load(self, is_test=False, prefix=None):
        if prefix is not None:
            transformer_path = self.transformer_path + prefix + '.pth'
        else:
            transformer_path = self.transformer_path + '_last.pth'
        if self.config.restore or is_test:
            if os.path.exists(transformer_path):
                print('Rank {} is loading {} Transformer...'.format(self.rank, transformer_path))
                data = torch.load(transformer_path, map_location="cpu")
                
                torch_init_model(self.img_encoder, transformer_path, 'img_encoder')
                torch_init_model(self.refine_module, transformer_path, 'refine')

                if self.config.restore:
                    self.opt.load_state_dict(data['opt'])
                    # skip sche
                    from tqdm import tqdm
                    for _ in tqdm(range(data['iteration']), desc='recover sche...'):
                        self.sche.step()
                self.iteration = data['iteration']
                self.sample_iter = data['sample_iter']
            else:
                print(transformer_path, 'not Found')
                raise FileNotFoundError

    
    def save(self, prefix=None):
        if prefix is not None:
            save_path = self.transformer_path + "_{}.pth".format(prefix)
        else:
            save_path = self.transformer_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        torch.save({
            'iteration': self.iteration,
            'sample_iter': self.sample_iter,
            'img_encoder': self.img_encoder.state_dict(),
            'refine': self.refine_module.state_dict(),
            'opt': self.opt.state_dict(),
        }, save_path)

        