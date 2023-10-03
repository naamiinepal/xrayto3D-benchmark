#!/usr/bin/bash
entity_name=msrepo
project_name=2d-3d-benchmark
trainpaths=configs/full/TotalSegmentor-ribs-DRR-full_train+val_patch.csv
testpaths=configs/full/TotalSegmentor-ribs-DRR-full_test_patch.csv
gpu=0
batch_size=4
lr=0.0002
steps=10000
img_size=128
res=1.25
loss=DiceLoss
#generate unique tag for this wandb run

unique_tag=$(python -c "import friendlywords as fw; print(fw.generate(1))")



# run python script and tag this wandb run 

CUDA_VISIBLE_DEVICES=1 python train.py  $trainpaths $testpaths --gpu $gpu --tags model-compare patch dropout $unique_tag --size $img_size --batch_size $batch_size --accelerator gpu --res $res --model_name CustomAutoEncoder --epochs -1 --loss $loss  --lr $lr  --dropout --steps $steps



# find the run-id with this unique wandb tag

run_id=$(wandb-utils -e $entity_name -p $project_name all-data --filters "{\"tags\":{\"\$in\":[\"$unique_tag\"]}}"  -f run  filter-df --pd-eval "df.run" print | tail -1 | cut -f 2)



# use the obtained run-id to find the latest checkpoint

CUDA_VISIBLE_DEVICES=1 python train.py  $trainpaths $testpaths --gpu $gpu --tags model-compare patch dropout --size $img_size --batch_size $batch_size --accelerator gpu --res $res --model_name TLPredictor --epochs -1 --loss $loss  --lr $lr  --dropout --load_autoencoder_from runs/2d-3d-benchmark/$run_id/checkpoints/last.ckpt --steps $steps
