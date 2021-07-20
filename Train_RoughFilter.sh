#!/bin/bash
#SBATCH -p gpu3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -o Train_RoughFilter_%j.out

conda activate tensorflow_1.14
module load matlab/R2018b
srun python3 -u train_phase.py \
--cover_dir /public/qinchuan/data/BOSS_BOWS2/256/train_cover_14000/ \
--stego_dir /public/qinchuan/data/BOSS_BOWS2/256/train_SUNIWARD_14000/payload_0.4/ \
--adv_dir /data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB-Pytorch/SiaStegNet/SUNIWARD/payload_0.4 \
--cover_feature_path /data-x/g15/qinchuan/feature/Spatial/imresize-256/SRM/cover.mat \
--stego_feature_path /data-x/g15/qinchuan/feature/Spatial/imresize-256/SRM/S-UNIWARD_0.4.mat \
--adv_feature_path /data-x/g15/qinchuan/feature/Spatial/imresize-256/SRM/ADV-EMB-Pytorch_SiaStegNet_SUNIWARD_payload_0.4.mat \
--ckpt_dir /public/qinchuan/deep-learning/SiaStegNet/SiaStegNet_BOSS_BOWS_SUNIWARD_payload_0.4.pth \
--clf_path /public/qinchuan/data/BOSS_BOWS2/256/ensemble_classifier/fixed_SRM_SUNIWARD_0.4.mat \
--kernel rbf --gamma 50 --C 20 \
--rough_filter_path ./Rough_Filter/SiaStegNet/SUNIWARD/payload_0.4/ \
--num_workers 4
date