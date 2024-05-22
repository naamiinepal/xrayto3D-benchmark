# Benchmark Results

This folder contains benchmark results for Xrayto3D architectures. We report the Dice Score, Haussdorff Distance, Normalized Surface Distance and Average Surface Distance（additionally, morphometric landmarks and distances are also reported）. Each setting runs a single seed and computes ranking stability[Wiesenfarth et. al, Scientific Reports, 2021].

M. Wiesenfarth, A. Reinke, B. A. Landman, M. Eisenmann, L. A. Saiz, M. J. Cardoso, L. Maier-Hein,
and A. Kopp-Schneider. Methods and open-source toolkit for analyzing and visualizing challenge results.
Scientific reports, 11(1):2369, 2021.

## Reproduction on Public Datasets Benchmark

### Datasets

| Dataset   | #Samples       | #Voxel Resolution (mm)| #Volume size | #Train/Test |
| --------- | ------------ | -------------- | ---------- | ------ |
| TotalSegmentor-Rib | 481 | 2.5        | 128^3     | 408/73     |
| TotalSegmentor-Femur | 465 | 1.0        | 128^3     | 786/138     |
| TotalSegmentor-Pelvic | 446 | 2.25        | 128^3      | 320/56     |
| CTPelvic1K |1106| 2.25         | 128^3     | 1106    |
| VerSe2019     | 1722 | 1.5      | 64^3    | 1451/271    |
| RSNACervicalFracture     | 710 | 1.5        | 64^3     | 710     |




### Aggregate Architecture Ranking

| Archtiecture     | #Params  | #DSC↑ | #HD95↓ | #ASD↓ | #NSD↑
| ----------- | ------- | --------------- | ---------- | ------ | ------ |
|SwinUNETR| 62.2M| 79.27| 3.65| 0.86| 0.68|
|AttentionUnet| 1.5M| 78.92| 3.07| 0.84| 0.69|
|TwoDPermuteConcat| 1.2M| 78.08| 3.33| 0.91| 0.67|
|UNet| 1.2M| 77.27| 3.49| 1.00| 0.66|
|MultiScale2DPermuteConcat| 3.5M| 77.09| 4.16| 0.96| 0.65|
|UNETR| 96.2M| 74.20| 4.27| 1.14| 0.62|
|TLPredictor| 6.6M| 69.53| 4.70| 1.43| 0.54|
|OneDConcat| 40.6M| 69.16| 7.07| 1.53| 0.53|

### Results

Please see the subdirectories for anatomy-wise [ [Vertebra](benchmarking/vertebra/), [Femur](benchmarking/femur/), [Hip](benchmarking/hip/), [rib](benchmarking/rib/)], architecture-wise and sample-wise results.

The [experiment_log_dir](experiment_log_dir/) contains additional evaluation on tasks such as domain-shifts, angle-perturbation and morphometry estimation.