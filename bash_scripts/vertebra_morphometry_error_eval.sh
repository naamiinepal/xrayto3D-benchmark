# model type: SwinUNETR tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/u66dbc2b/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/u66dbc2b/evaluation/vertebra_morphometry.csv
# model type: UNETR tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/0ugb85wj/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/0ugb85wj/evaluation/vertebra_morphometry.csv
# model type: AttentionUnet tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/p3qkfyj5/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/p3qkfyj5/evaluation/vertebra_morphometry.csv
# model type: UNet tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/30wlxp31/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/30wlxp31/evaluation/vertebra_morphometry.csv
# model type: MultiScale2DPermuteConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/n4xympug/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/n4xympug/evaluation/vertebra_morphometry.csv
# model type: TwoDPermuteConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/e9y5hclj/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/e9y5hclj/evaluation/vertebra_morphometry.csv
# model type: OneDConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/armvulbx/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/armvulbx/evaluation/vertebra_morphometry.csv
# model type: TLPredictor tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/82esz36y/evaluation --log_filename vertebra_morphometry.csv
python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/82esz36y/evaluation/vertebra_morphometry.csv
