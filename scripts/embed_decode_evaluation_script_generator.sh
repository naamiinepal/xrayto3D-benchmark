python scripts/generate_embed_decode_arch_evaluation_script.py --testpaths configs/paths/verse19/Verse2019-DRR-full_test.csv --gpu 0 --batch_size 8 --img_size 64 --res 1.5 > bash_scripts/evaluate_verse19_dropout_embed_decode.sh

python scripts/generate_embed_decode_arch_evaluation_script.py --testpaths configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_test.csv  --gpu 0 --img_size 128 --res 2.5 --batch_size 8 > bash_scripts/evaluate_rib_dropout_embed_decode.sh

python scripts/generate_embed_decode_arch_evaluation_script.py  --testpaths configs/paths/totalsegmentator_hips/TotalSegmentor-hips-DRR-full_test.csv  --gpu 0 --img_size 128 --res 2.25 --batch_size 8 > bash_scripts/evaluate_totalseg_hip_dropout_embed_decode.sh

python scripts/generate_embed_decode_arch_evaluation_script.py  --testpaths configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv  --gpu 0 --img_size 128 --res 1.0 --batch_size 8 > bash_scripts/evaluate_totalseg_femur_dropout_embed_decode.sh