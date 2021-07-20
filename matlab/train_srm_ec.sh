#!/bin/bash

#SBATCH -p cpu4
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -o train_srm_ec_adv-emb_siastegnet_suniward_0.4-%j.out
#SBATCH --nodelist wmc-slave-g14

date
module load matlab
# 下面的 test_cpu.m 换成你的 MATLAB 脚本的路径
# for pl in 0.4; do
#     srun matlab -nodisplay -nosplash -nodesktop -r "clear;\
#     cover_dir = '/data-x/g15/qinchuan/Spatial/imresize-256/cover/';\
#     payload = ${pl};\
#     stego_dir = ['/data-x/g15/qinchuan/Spatial/imresize-256/S-UNIWARD/payload_', num2str(payload), '/'];\
#     feature_dir = '/data-x/g15/qinchuan/feature/Spatial/imresize-256/SRM/';\
#     ec_path = ['~/data/BOSS_BOWS2/256/ensemble_classifier/fixed_SRM_SUNIWARD_', num2str(payload),'.mat'];\
#     ref_trn_dir = '~/data/BOSS_BOWS2/256/train_cover_14000/'; ref_val_dir = '~/data/BOSS_BOWS2/256/val_cover_1000/'; ref_tst_dir = '~/data/BOSS_BOWS2/256/test_cover_5000/';\
#     TRN_TST_EC(cover_dir, stego_dir, feature_dir, payload, ec_path, ref_trn_dir, ref_val_dir, ref_tst_dir);\
#     exit;"
# done
# date
for pl in 0.4; do
    srun matlab -nodisplay -nosplash -nodesktop -r "clear;\
    cover_dir = '/data-x/g15/qinchuan/Spatial/imresize-256/cover/';\
    payload = ${pl};\
    stego_dir = ['/data-x/g15/qinchuan/Spatial/imresize-256/ADV-EMB-Pytorch/SiaStegNet/SUNIWARD/payload_', num2str(payload), '/'];\
    feature_dir = '/data-x/g15/qinchuan/feature/Spatial/imresize-256/SRM/';\
    ec_path = ['~/data/BOSS_BOWS2/256/ensemble_classifier/fixed_SRM_ADV-EMB_SiaStegNet_SUNIWARD_', num2str(payload),'.mat'];\
    ref_trn_dir = '~/data/BOSS_BOWS2/256/train_cover_14000/'; ref_val_dir = '~/data/BOSS_BOWS2/256/val_cover_1000/'; ref_tst_dir = '~/data/BOSS_BOWS2/256/test_cover_5000/';\
    TRN_TST_EC(cover_dir, stego_dir, feature_dir, payload, ec_path, ref_trn_dir, ref_val_dir, ref_tst_dir);\
    exit;"
done
date