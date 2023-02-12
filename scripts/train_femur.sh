#!/usr/bin/bash

# gpu 1 is TITAN XP
# gpu 0 is RTX 3090

# train unet
python train_unetv2.py configs/paths/femur/50k/TotalSegmentor-femur-left-DRR-full_train.csv configs/paths/femur/50k/TotalSegmentor-femur-left-DRR-full_val.csv --gpu 1 --tags full femur --size 128 --batch_size 4 --accelerator gpu --res 1.0

python train_unetv2.py configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_train.csv configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_val.csv --gpu 1 --tags full femur --size 128 --batch_size 4 --accelerator gpu --res 1.0

# evaluate unet
python train_unetv2.py configs/full/TotalSegmentor-femur-left-DRR-full_train.csv configs/full/fullpaths/totalsegmentator_femur_left/TotalSegmentor-femur-left-DRR-full_val.csv --gpu 1 --tags full femur --size 128 --batch_size 4 --accelerator gpu --res 1.0 --evaluate --save_predictions --checkpoint_path pipeline-test-01/zygidk3d/checkpoints/epoch=26-step=2213.ckpt --output_dir pipeline-test-01/zygidk3d/out0

#train 1dconcat
python train_1dconcat_v2.py configs/full/TotalSegmentor-femur-left-DRR-full_train.csv configs/full/fullpaths/totalsegmentator_femur_left/TotalSegmentor-femur-left-DRR-full_val.csv --gpu 1 --tags full femur --size 128 --batch_size 4 --accelerator gpu --res 1.0