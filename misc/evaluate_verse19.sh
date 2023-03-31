python evaluate.py  --testpaths configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 8 --accelerator gpu --res 1.5 --model_name OneDConcat --ckpt_path runs/2d-3d-benchmark/929yh5ep/checkpoints --gpu 0

python evaluate.py  --testpaths configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 8 --accelerator gpu --res 1.5 --model_name MultiScale2DPermuteConcat --ckpt_path runs/2d-3d-benchmark/spnv4lq0/checkpoints --gpu 0

python evaluate.py  --testpaths configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 8 --accelerator gpu --res 1.5 --model_name TwoDPermuteConcat --ckpt_path runs/2d-3d-benchmark/m5be5wrr/checkpoints --gpu 0

python evaluate.py  --testpaths configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 8 --accelerator gpu --res 1.5 --model_name AttentionUnet --ckpt_path runs/2d-3d-benchmark/w8rczz8u/checkpoints --gpu 0

python evaluate.py  --testpaths configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 8 --accelerator gpu --res 1.5 --model_name UNet --ckpt_path runs/2d-3d-benchmark/30wlxp31/checkpoints --gpu 0

