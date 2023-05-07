swinunetr_run_id=$(wandb-utils -e msrepo -p $project_name all-data --filters "{\"\$and\":[{\"tags\":{\"\$in\":[\"model-compare\"]}},{\"tags\":{\"\$in\":[\"SwinUNETR\"]}},{\"tags\":{\"\$in\":[\"$anatomy\"]}}]}"   -f run  filter-df --pd-eval "df.run" print | tail -1 | cut -f 2)

python evaluate.py --testpaths $testpaths --gpu $gpu --image_size $img_size --batch_size $batch_size --accelerator gpu --res $res --model_name SwinUNETR --ckpt_path runs/$project_name/$swinunetr_run_id/checkpoints
