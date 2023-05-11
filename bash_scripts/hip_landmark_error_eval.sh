# model type: SwinUNETR tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/gzekjp1r/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/gzekjp1r/evaluation/hip_landmarks.csv
# model type: UNETR tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/762ji1eb/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/762ji1eb/evaluation/hip_landmarks.csv
# model type: AttentionUnet tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/yiw2kgep/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/yiw2kgep/evaluation/hip_landmarks.csv
# model type: UNet tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/ktkdfd5v/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/ktkdfd5v/evaluation/hip_landmarks.csv
# model type: MultiScale2DPermuteConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/dnnwydzk/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/dnnwydzk/evaluation/hip_landmarks.csv
# model type: TwoDPermuteConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/y8kln4px/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/y8kln4px/evaluation/hip_landmarks.csv
# model type: OneDConcat tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/hw4es5nw/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/hw4es5nw/evaluation/hip_landmarks.csv
# model type: TLPredictor tags ['dropout', 'model-compare']
python external/xrayto3D-morphometry/hip_landmarks_v2.py --dir runs/2d-3d-benchmark/r3cysgm6/evaluation --log_filename hip_landmarks.csv
python external/xrayto3D-morphometry/process_hip_csv.py runs/2d-3d-benchmark/r3cysgm6/evaluation/hip_landmarks.csv
