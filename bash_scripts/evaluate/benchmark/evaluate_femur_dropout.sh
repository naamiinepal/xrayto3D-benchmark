python evaluate.py  --testpaths configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 1.0 --model_name OneDConcat --ckpt_path runs/2d-3d-benchmark/j9mkkkxc/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/j9mkkkxc/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 1.0 --model_name MultiScale2DPermuteConcat --ckpt_path runs/2d-3d-benchmark/2qpdmdk9/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/2qpdmdk9/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 1.0 --model_name TwoDPermuteConcat --ckpt_path runs/2d-3d-benchmark/fs8n9l5j/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/fs8n9l5j/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 1.0 --model_name AttentionUnet --ckpt_path runs/2d-3d-benchmark/nmsxz4z4/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/nmsxz4z4/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 1.0 --model_name UNet --ckpt_path runs/2d-3d-benchmark/nuhm8qx9/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/nuhm8qx9/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 1.0 --model_name UNETR --ckpt_path runs/2d-3d-benchmark/27uay0sp/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/27uay0sp/checkpoints/../evaluation

