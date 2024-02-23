CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset KINS --batch 1 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset KINS --batch 32 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 128 --data_type image --path UOAIS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_c2f_seg.py --dataset UOAIS --batch 64 --data_type image --path UOAIS_c2f_seg
###### 上面的用前都需要check #####

# test original
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg --model original

# train depth
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg_4channel --model rgbd_4channel
