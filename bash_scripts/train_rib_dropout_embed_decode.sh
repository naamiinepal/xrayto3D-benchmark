#!/usr/bin/bash
entity_name=msrepo
project_name=2d-3d-benchmark
trainpaths=configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_train+val.csv
testpaths=configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv
gpu=0
batch_size=8
lr=0.0002
steps=5000
tag=model-compare
img_size=128
res=2.5
loss=DiceLoss
#generate unique tag for this wandb run

unique_tag=$(python -c "import friendlywords as fw; print(fw.generate(1))")



# run python script and tag this wandb run 

python train.py  $trainpaths $testpaths --gpu $gpu --tags $tag $unique_tag --size $img_size --batch_size $batch_size --accelerator gpu --res $res --model_name CustomAutoEncoder --epochs -1 --loss $loss  --lr $lr  --dropout --steps $steps



# find the run-id with this unique wandb tag

run_id=$(wandb-utils -e $entity_name -p $project_name all-data --filters "{\"tags\":{\"\$in\":[\"$unique_tag\"]}}"  -f run  filter-df --pd-eval "df.run" print | tail -1 | cut -f 2)



# use the obtained run-id to find the latest checkpoint

python train.py  $trainpaths $testpaths --gpu $gpu --tags $tag --size $img_size --batch_size $batch_size --accelerator gpu --res $res --model_name TLPredictor --epochs -1 --loss $loss  --lr $lr  --dropout --load_autoencoder_from runs/2d-3d-benchmark/$run_id/checkpoints/last.ckpt --steps $steps
