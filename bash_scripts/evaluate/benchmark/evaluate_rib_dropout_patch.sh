# python evaluate.py  --testpaths configs/full/TotalSegmentor-ribs-DRR-full_test_patch.csv --gpu 0 --image_size 128 --batch_size 16 --accelerator gpu --res 1.25 --model_name OneDConcat --ckpt_path runs/2d-3d-benchmark/sg2b7t70/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/sg2b7t70/checkpoints/../evaluation

python evaluate.py  --testpaths configs/full/TotalSegmentor-ribs-DRR-full_test_patch.csv --gpu 0 --image_size 128 --batch_size 16 --accelerator gpu --res 1.25 --model_name MultiScale2DPermuteConcat --ckpt_path runs/2d-3d-benchmark/ll0b15mt/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/ll0b15mt/checkpoints/../evaluation

python evaluate.py  --testpaths configs/full/TotalSegmentor-ribs-DRR-full_test_patch.csv --gpu 0 --image_size 128 --batch_size 16 --accelerator gpu --res 1.25 --model_name AttentionUnet --ckpt_path runs/2d-3d-benchmark/ppu4169g/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/ppu4169g/checkpoints/../evaluation

python evaluate.py  --testpaths configs/full/TotalSegmentor-ribs-DRR-full_test_patch.csv --gpu 0 --image_size 128 --batch_size 16 --accelerator gpu --res 1.25 --model_name UNet --ckpt_path runs/2d-3d-benchmark/wwm8cp9h/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/wwm8cp9h/checkpoints/../evaluation

python evaluate.py  --testpaths configs/full/TotalSegmentor-ribs-DRR-full_test_patch.csv --gpu 0 --image_size 128 --batch_size 16 --accelerator gpu --res 1.25 --model_name UNETR --ckpt_path runs/2d-3d-benchmark/kl7xzdt8/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/kl7xzdt8/checkpoints/../evaluation

python evaluate.py  --testpaths configs/full/TotalSegmentor-ribs-DRR-full_test_patch.csv --gpu 0 --image_size 128 --batch_size 16 --accelerator gpu --res 1.25 --model_name SwinUNETR --ckpt_path runs/2d-3d-benchmark/vs1e9hxu/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/vs1e9hxu/checkpoints/../evaluation

