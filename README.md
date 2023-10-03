# xrayto3D-benchmark
This is a pytorch-based Python package for Biplanar X-ray to 3D Shape Segmentation. We aim to become a comprehensive benchmarking tool for developing and evaluating architectures
specific to this problem. Currently, we provide tools to train and evaluate on 4 different bone anatomies, using publicly available CT-segmentation datasets. We also define tasks
for domain shifts to evaluate the robustness of these methods. Currently, various Encoder-Decoder Architectures on volumetric grid-based representation are reimplemented and benchmarked.

![benchmark framework](https://github.com/naamiinepal/xrayto3D-benchmark/assets/10219364/caeffd06-bdb1-40d6-973d-4a19540fe885)
![benchmark_overview_updated svg](https://github.com/naamiinepal/xrayto3D-benchmark/assets/10219364/e263b5d8-f5f0-46fe-baf5-5bd1040f0628)


#### Getting Started
---
##### Prerequisites
To install the required packages
```bash
conda create env --name xrayto3dbenchmark-cuda_11 --f  benchmark-requirements-cuda_11.yaml
# or
conda create env --name xrayto3dbenchmark-cuda_10 --f benchmark-requirements-cuda_10.yaml
conda activate ...
```
Install the XrayTo3DShape package
```bash
pip install XrayTo3DShape  
```
Sample Training script call
```sh
python train.py  configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_train+val.csv configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv --gpu 0 --tags model-compare --size 128 --batch_size 4 --accelerator gpu --res 1.0 --model_name MultiScale2DPermuteConcat --epochs -1 --loss DiceLoss  --lr 0.002 --steps 4000 --dropout
```

#### Prepare Datasets
A separate repo for preprocessing data has been created [here](https://github.com/naamiinepal/XrayTo3DPreprocess). The instructions for downloading and processing datasets are described [here.](https://github.com/naamiinepal/XrayTo3DPreprocess/blob/main/workflow/readme.md)

#### Usage
---
Examples to help you get familiar with the XrayTo3DShape package for quick use, evaluate an existing architecture on your own dataset, or benchmark new architectures.

##### Quick Start
- Beginning Examples
  - [Sample Model Prediction](docs/sample_evaluation.ipynb)
- [Customize Datasets](docs/dataset.md)
- [Regarding Orientation alignment between Input X-rays and Output Segmentation](docs/orientation.md)

#### Training
See `bash_scripts` dir

#### Evaluation
- [Domain Shift](docs/Domain_Shift_Evaluation.md)
- [Angle Perturbation](docs/Angle_Perturbation_evaluation.md)
#### Benchmark Results
[wandb results page](https://wandb.ai/msrepo/2d-3d-benchmark)

#### Model Zoo
TODO: Add pre-trained models

#### Licence
The code is made available under the MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. 
The manuscript text (including all LaTeX files), figures, and models produced as part of this research are available under the Creative Commons Attribution 4.0 License (CC-BY). 

#### Acknowledgements
We took considerable inspiration and references from:
- [USB](https://github.com/microsoft/Semi-supervised-learning)
- [MONAI](github.com/project-MONAI/MONAI)

Various adaptation of code(or code snippets) from these sources:
- [FracReconNet](https://github.com/DanupongBu/FracReconNet)
- [VerSe](https://github.com/anjany/verse)
- [torch-template](https://github.com/shagunsodhani/torch-template)

Various sections of the code uses following external codebases:
- [SpatialConfiguration-Net](https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe)
- [surface-distance](https://github.com/deepmind/surface-distance)
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
