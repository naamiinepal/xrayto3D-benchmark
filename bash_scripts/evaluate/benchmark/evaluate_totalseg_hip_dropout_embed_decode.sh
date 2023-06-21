#!/usr/bin/bash
entity_name=msrepo
project_name=2d-3d-benchmark
testpaths=configs/paths/totalsegmentator_hips/TotalSegmentor-hips-DRR-full_test.csv
anatomy=hip
img_size=128
res=2.25
gpu=0
batch_size=8
# find the run-id for the corresponding CustomAutoencoder for the TLPredictors.



autoencoder_run_id=$(wandb-utils -e $entity_name -p $project_name all-data --filters "{\"\$and\":[{\"tags\":{\"\$in\":[\"model-compare\"]}},{\"tags\":{\"\$in\":[\"CustomAutoEncoder\"]}},{\"tags\":{\"\$in\":[\"$anatomy\"]}}]}"   -f run  filter-df --pd-eval "df.run" print | tail -1 | cut -f 2)



tlencoder_run_id=$(wandb-utils -e msrepo -p $project_name all-data --filters "{\"\$and\":[{\"tags\":{\"\$in\":[\"model-compare\"]}},{\"tags\":{\"\$in\":[\"TLPredictor\"]}},{\"tags\":{\"\$in\":[\"$anatomy\"]}}]}"   -f run  filter-df --pd-eval "df.run" print | tail -1 | cut -f 2)



python evaluate.py --testpaths $testpaths --gpu $gpu --image_size $img_size --batch_size $batch_size --accelerator gpu --res $res --model_name TLPredictor --ckpt_path runs/$project_name/$tlencoder_run_id/checkpoints --load_autoencoder_from runs/$project_name/$autoencoder_run_id/checkpoints/last.ckpt
