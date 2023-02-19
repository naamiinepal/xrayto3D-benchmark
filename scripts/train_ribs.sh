#!/usr/bin/bash
python train_2dpermuteconcat_v2.py configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_train.csv  configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_val.csv --gpu 0 --tags full ribs --size 128 --batch_size 2 --accelerator gpu --res 2.5 --precision 16

python train_1dconcat_v2.py configs/full/fullpaths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_train.csv  configs/full/fullpaths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_val.csv --gpu 0 --tags full ribs --size 128 --batch_size 2 --accelerator gpu --res 2.5 --precision 16