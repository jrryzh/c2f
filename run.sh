CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset KINS --batch 1 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset KINS --batch 32 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg