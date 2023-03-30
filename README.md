<h1 align="center">BEVSimDet: Simulated Multi-modal Distillation in Bird’s-Eye View for Multi-view 3D Object Detection</h1>
<p align="center">
<a href="https://arxiv.org/abs/2303.16818"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2303.16818">BEVSimDet: Simulated Multi-modal Distillation in Bird’s-Eye View for Multi-view 3D Object Detection</a>.</h4>
<h5 align="center"><em>Haimei Zhao, Qiming Zhang, Shanshan Zhao, Jing Zhang, and Dacheng Tao</em></h5>
<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#method">Method</a> |
  <a href="#results">Results</a> |
  <a href="#preparation">Preparation</a> |
  <a href="#code">Code</a> |
  <a href="#statement">Statement</a>
</p>

## News

- **(2023/3/29)** BEVFusion is released on [arXiv](https://arxiv.org/abs/2303.16818).

## Abstract

Multi-view camera-based 3D object detection has gained popularity due to its low cost. But accurately inferring 3D geometry solely from camera data remains challenging, which impacts model performance. One promising approach to address this issue is to distill precise 3D geometry knowledge from LiDAR data. However, transferring knowledge between different sensor modalities is hindered by the significant modality gap. In this paper, we approach this challenge from the perspective of both architecture design and knowledge distillation and present a new simulated multi-modal 3D object detection method named BEVSimDet. We first introduce a novel framework that includes a LiDAR and camera fusion-based teacher and a simulated multi-modal student, where the student simulates multi-modal features with image-only input. To facilitate effective distillation, we propose a simulated multi-modal distillation scheme that supports intra-modal, cross-modal, and multi-modal distillation simultaneously. By combining them together, BEVSimDet can learn better feature representations for 3D object detection while enjoying cost-effective camera-only deployment. Experimental results on the challenging nuScenes benchmark demonstrate the effectiveness and superiority of BEVSimDet over recent representative methods.
## Method

![the framework figure](./figs/mainfigure.png "framework")
## Results

### Quantitative results on Nuscenes validation set
![quantitative figure](./figs/quantitative-results.png "quantitative-results")
### Qualitative results
![qualitative figure](./figs/visualization.png "visualization")
![qualitative figure](./figs/supplementary-lidar.png "supplementary-lidar")
![qualitative figure](./figs/supplementary-prediction1.png "supplementary-prediction1")
## Preparation

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```
### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

## Code
### Codes coming soon...

## Statement

