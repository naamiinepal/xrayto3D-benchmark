python evaluate.py  --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 2 --accelerator gpu --res 1.5 --model_name OneDConcat --ckpt_path runs/2d-3d-benchmark/armvulbx/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/armvulbx/checkpoints/../domain_shift_rsna

python evaluate.py  --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 2 --accelerator gpu --res 1.5 --model_name MultiScale2DPermuteConcat --ckpt_path runs/2d-3d-benchmark/n4xympug/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/n4xympug/checkpoints/../domain_shift_rsna

python evaluate.py  --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 2 --accelerator gpu --res 1.5 --model_name TwoDPermuteConcat --ckpt_path runs/2d-3d-benchmark/e9y5hclj/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/e9y5hclj/checkpoints/../domain_shift_rsna

python evaluate.py  --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 2 --accelerator gpu --res 1.5 --model_name AttentionUnet --ckpt_path runs/2d-3d-benchmark/p3qkfyj5/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/p3qkfyj5/checkpoints/../domain_shift_rsna

python evaluate.py  --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 2 --accelerator gpu --res 1.5 --model_name UNet --ckpt_path runs/2d-3d-benchmark/30wlxp31/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/30wlxp31/checkpoints/../domain_shift_rsna

python evaluate.py  --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 2 --accelerator gpu --res 1.5 --model_name UNETR --ckpt_path runs/2d-3d-benchmark/0ugb85wj/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/0ugb85wj/checkpoints/../domain_shift_rsna

python evaluate.py  --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv --gpu 0 --image_size 64 --batch_size 2 --accelerator gpu --res 1.5 --model_name SwinUNETR --ckpt_path runs/2d-3d-benchmark/u66dbc2b/checkpoints --ckpt_type latest --gpu 0 --output_path runs/2d-3d-benchmark/u66dbc2b/checkpoints/../domain_shift_rsna

