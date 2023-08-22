python scripts/generate_embed_decode_arch_domainshift_script.py --testpaths configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv   --gpu 0 --batch_size 8 --img_size 64 --res 1.5 --domain_shift_dataset rsna> bash_scripts/evaluate/domainshift/evaluate_vertebra_rsna__dropout_embed_decode.sh

python scripts/generate_embed_decode_arch_domainshift_script.py --testpaths configs/domain_shift_eval/CTPelvic1k_abdomen-hips-DRR-full.csv  --gpu 0 --batch_size 8 --img_size 128 --res 2.25 --domain_shift_dataset abdomen> bash_scripts/evaluate/domainshift/evaluate_CTPelvic1k_abdomen_hips_dropout_embed_decode.sh

python scripts/generate_embed_decode_arch_domainshift_script.py --testpaths configs/domain_shift_eval/CTPelvic1k_CLINIC-hips-DRR-full.csv   --gpu 0 --batch_size 8 --img_size 128 --res 2.25 --domain_shift_dataset clinic> bash_scripts/evaluate/domainshift/evaluate_CTPelvic1k_clinic_hips_dropout_embed_decode.sh

python scripts/generate_embed_decode_arch_domainshift_script.py --testpaths configs/domain_shift_eval/CTPelvic1k_CLINIC-METAL-hips-DRR-full.csv   --gpu 0 --batch_size 8 --img_size 128 --res 2.25 --domain_shift_dataset clinic_metal> bash_scripts/evaluate/domainshift/evaluate_CTPelvic1k_clinic_metal_hips_dropout_embed_decode.sh

python scripts/generate_embed_decode_arch_domainshift_script.py --testpaths configs/domain_shift_eval/CTPelvic1k_kits-hips-DRR-full.csv   --gpu 0 --batch_size 8 --img_size 128 --res 2.25 --domain_shift_dataset kits> bash_scripts/evaluate/domainshift/evaluate_CTPelvic1k_kits_hips_dropout_embed_decode.sh 