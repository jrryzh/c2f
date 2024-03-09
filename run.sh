CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset KINS --batch 1 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset KINS --batch 32 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 128 --data_type image --path UOAIS_c2f_seg
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_c2f_seg.py --dataset UOAIS --batch 64 --data_type image --path UOAIS_c2f_seg
###### 上面的用前都需要check #####

# train original
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg --model original

# test original
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD --batch 32 --data_type image --path UOAIS_c2f_seg --model original
# allvm
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg --model original
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg --model original


# train depth_only
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg_depth --model depth_only

# test depth_only
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS --batch 16 --data_type image --path UOAIS_c2f_seg_depth --model depth_only
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD --batch 16 --data_type image --path UOAIS_c2f_seg_depth --model depth_only
# allvm
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_depth --model depth_only
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_depth --model depth_only

# train depth_6channel
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg_6channel --model rgbd_6channel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=7 train_c2f_seg.py --dataset UOAIS --batch 16 --data_type image --path UOAIS_c2f_seg_6channel --model rgbd_6channel

# test depth_6channel
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS --batch 16 --data_type image --path UOAIS_c2f_seg_6channel --model rgbd_6channel
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD --batch 16 --data_type image --path UOAIS_c2f_seg_6channel --model rgbd_6channel
# allvm
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_6channel --model rgbd_6channel
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_6channel --model rgbd_6channel

# train depth_fusion
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg_fusion --model rgbd_fusion
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=7 train_c2f_seg.py --dataset UOAIS --batch 16 --data_type image --path UOAIS_c2f_seg_fusion --model rgbd_fusion

# test depth_fusion
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS --batch 16 --data_type image --path UOAIS_c2f_seg_fusion --model rgbd_fusion
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD --batch 16 --data_type image --path OSD_c2f_seg_fusion --model rgbd_fusion
# allvm
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_fusion --model rgbd_fusion
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_fusion --model rgbd_fusion

# train depth_linearfusion
CUDA_VISIBLE_DEVICES=0 python train_c2f_seg_1gpu.py --dataset UOAIS --batch 32 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion

# test depth_linearfusion
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS --batch 16 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD --batch 16 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion
# allvm
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset UOAIS_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu.py --dataset OSD_ALLVM --batch 32 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion


## eval metrics
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu_amodal_metrics.py --dataset OSD --batch 32 --data_type image --path UOAIS_c2f_seg --model original
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu_amodal_metrics.py --dataset OSD --batch 32 --data_type image --path UOAIS_c2f_seg_6channel --model rgbd_6channel
CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu_amodal_metrics.py --dataset OSD --batch 32 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion

CUDA_VISIBLE_DEVICES=0 python test_c2f_seg_1gpu_amodal_metrics_uoaisvm.py --dataset OSD --batch 32 --data_type image --path UOAIS_c2f_seg_linearfusion --model rgbd_linearfusion