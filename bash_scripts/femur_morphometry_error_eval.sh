# model type: SwinUNETR tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/0hoo4r74/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/0hoo4r74/evaluation/femur_morphometry.csv
# model type: UNETR tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/27uay0sp/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/27uay0sp/evaluation/femur_morphometry.csv
# model type: AttentionUnet tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/nmsxz4z4/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/nmsxz4z4/evaluation/femur_morphometry.csv
# model type: UNet tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/nuhm8qx9/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/nuhm8qx9/evaluation/femur_morphometry.csv
# model type: MultiScale2DPermuteConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/2qpdmdk9/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/2qpdmdk9/evaluation/femur_morphometry.csv
# model type: TwoDPermuteConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/fs8n9l5j/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/fs8n9l5j/evaluation/femur_morphometry.csv
# model type: OneDConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/j9mkkkxc/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/j9mkkkxc/evaluation/femur_morphometry.csv
# model type: TLPredictor tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/femur_morphometry_v2.py --dir runs/2d-3d-benchmark/s5gzofmq/evaluation --log_filename femur_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_femur_csv.py runs/2d-3d-benchmark/s5gzofmq/evaluation/femur_morphometry.csv
