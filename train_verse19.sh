python train.py  configs/paths/verse19/Verse2019-DRR-full_train+val.csv configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --tags model-compare --size 96 --batch_size 8 --accelerator gpu --res 1.0 --model_name OneDConcat --epochs -1 --loss DiceLoss  --lr 0.0002 --steps 15000 

python train.py  configs/paths/verse19/Verse2019-DRR-full_train+val.csv configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --tags model-compare --size 96 --batch_size 8 --accelerator gpu --res 1.0 --model_name MultiScale2DPermuteConcat --epochs -1 --loss DiceLoss  --lr 0.0002 --steps 15000 

python train.py  configs/paths/verse19/Verse2019-DRR-full_train+val.csv configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --tags model-compare --size 96 --batch_size 8 --accelerator gpu --res 1.0 --model_name TwoDPermuteConcat --epochs -1 --loss DiceLoss  --lr 0.0002 --steps 15000 

python train.py  configs/paths/verse19/Verse2019-DRR-full_train+val.csv configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --tags model-compare --size 96 --batch_size 8 --accelerator gpu --res 1.0 --model_name AttentionUnet --epochs -1 --loss DiceLoss  --lr 0.0002 --steps 15000 

python train.py  configs/paths/verse19/Verse2019-DRR-full_train+val.csv configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --tags model-compare --size 96 --batch_size 8 --accelerator gpu --res 1.0 --model_name UNet --epochs -1 --loss DiceLoss  --lr 0.0002 --steps 15000 

