python evaluate.py  --testpaths configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 2.5 --model_name OneDConcat --ckpt_path runs/2d-3d-benchmark/ab1i0a5e/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/ab1i0a5e/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 2.5 --model_name MultiScale2DPermuteConcat --ckpt_path runs/2d-3d-benchmark/35g59veo/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/35g59veo/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 2.5 --model_name TwoDPermuteConcat --ckpt_path runs/2d-3d-benchmark/10hy7jzr/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/10hy7jzr/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 2.5 --model_name AttentionUnet --ckpt_path runs/2d-3d-benchmark/irc6n7mq/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/irc6n7mq/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 2.5 --model_name UNet --ckpt_path runs/2d-3d-benchmark/64xxm1gj/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/64xxm1gj/checkpoints/../evaluation

python evaluate.py  --testpaths configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv --gpu 0 --image_size 128 --batch_size 8 --accelerator gpu --res 2.5 --model_name UNETR --ckpt_path runs/2d-3d-benchmark/pwficftp/checkpoints --gpu 0 --output_path runs/2d-3d-benchmark/pwficftp/checkpoints/../evaluation

