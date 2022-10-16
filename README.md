<div align="center">

# Benchmarking 3D Pose and Shape Estimation Beyond Algorithms

<div>
    <a href='' target='_blank'>Hui En Pang</a>&emsp;
    <a href='https://caizhongang.github.io/' target='_blank'>Zhongang Cai</a>&emsp;
    <a href='https://yanglei.me/' target='_blank'>Lei Yang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=9vpiYDIAAAAJ&hl=en' target='_blank'>Tianwei Zhang</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>*</sup>
</div>
<div>
    S-Lab, Nanyang Technological University&emsp; <sup>*</sup>corresponding author
</div>


<h4 align="center">
  <a href="https://arxiv.org/abs/2209.10529" target='_blank'>[arXiv]</a> •
  <a href="" target='_blank'>[Slides]</a>
</h4>

## Getting started
### [Installation](#installation) | [Train](#train) | [Evaluation](#evaluation) | [FLOPs](#flops) |

## Experiments
### [Single-datasets](#single-datasets) | [Mixed-datasets](#mixed-datasets) | [Augmentations](#augmentations) | [Backbones](#backbones) | [Losses](#losses) | [Backbone-initialisation](#backbone-initialisation) | [Algorithms](#algorithms) | [Downloads](#downloads) |

</div>

## Introduction

This repository builds upon [MMHuman3D](https://openmmlab.com/mmhuman3d), an open source PyTorch-based codebase for the use of 3D human parametric models in computer vision and computer graphics. MMHuman3D is a part of the [OpenMMLab](https://openmmlab.com/) project. The main branch works with **PyTorch 1.7+**.

These features will be contributed to MMHuman3D at a later date.

<!--
https://user-images.githubusercontent.com/62529255/144362861-e794b404-c48f-4ebe-b4de-b91c3fbbaa3b.mp4 -->


<!-- HMR+ uses the same ResNet-50 backbone and training datasets as HMR and SPIN (without fittings). We find that adopting our training tricks is sufficient to build a model that is comparable to the previous SOTA algorithms without using bigger model capacity or newer datasets. We also include HMR* uses Twins-SVT backbone and datasets following PARE.  See our [paper]() for more details. -->
<p align="center">
    <!-- <img src="resources/dance3.gif" width="99%"> -->
    <img src="resources/dance.gif" width="99%">
    <img src="resources/dance001.gif" width="80%">
</p>

### Major Features added to MMHuman3D

We have added multiple major features on top of MMHuman3D.
- **Benchmarks on 31 datasets**
- **Benchmarks on 11 dataset combinations**
- **Benchmarks on 9 backbones and different initialisation**
- **Benchmarks on 9 augmentation techniques**
- **Provide trained models on optimal configurations for inference**
- **Evaluation on 5 test sets**
- **FLOPs calculation**

Additional:
- Train annotation files for 31 datasets will be provided in the future
- Future works can easily obtain benchmarks on HMR for baseline comparison on their selected dataset mixes and partition using our provided pipeline and annotation files.

<!-- #### [Single-datasets](#single-datasets) | [Mixed--datasets](#installation) | [Train](#train) | [Evaluation](#evaluation) | [FLOPs](#flops) | -->


<!-- - **Provide annotation files**

  Will add links to Gdrive. Future works can use these files to train on any of the 31 dataset. -->
<!-- - **Provide config files for 31 single and multi-dataset benchmarks** -->
## Experiments
### Single-datasets
<!--
  Future works can use these configs for training and obtain benchmarks on HMR for baseline comparison on their selected dataset mixes and partition. -->

Supported datasets:
<details open>
<summary>(click to collapse)</summary>

1. [AGORA](https://agora.is.tue.mpg.de/) (CVPR'2021)
2. [AI Challenger](https://challenger.ai/) (ICME'2019)
3. [COCO](https://cocodataset.org/#home) (ECCV'2014)
4. [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) (ECCV'2020)
5. [EFT-COCO-Part](https://github.com/facebookresearch/eft) (3DV'2021)
6. [EFT-COCO](https://github.com/facebookresearch/eft) (3DV'2021)
7. [EFT-LSPET](https://github.com/facebookresearch/eft) (3DV'2021)
8. [EFT-OCHuman](https://github.com/facebookresearch/eft) (3DV'2021)
9. [EFT-PoseTrack](https://github.com/facebookresearch/eft) (3DV'2021)
10. [EFT-MPII](https://github.com/facebookresearch/eft) (3DV'2021)
11. [Human3.6M](http://vision.imar.ro/human3.6m/description.php) (TPAMI'2014)
12. [InstaVariety](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md) (CVPR'2019)
13. [LIP](https://www.sysu-hcp.net/projects/cv/38.html) (CVPR'2017)
14. [LSP](https://sam.johnson.io/research/lsp.html) (BMVC'2010)
15. [LSP-Extended](https://sam.johnson.io/research/lspet.html) (CVPR'2011)
16. [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) (3DC'2017)
17. [MPII](http://human-pose.mpi-inf.mpg.de/) (CVPR'2014)
18. [MTP](https://tuch.is.tue.mpg.de/) (CVPR'2021)
19. [MuCo-3DHP](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) (3DV'2018)
20. [MuPoTs-3D](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) (3DV'2018)
21. [OCHuman](http://www.liruilong.cn/project_pages/pose2seg.html) (CVPR'2019)
22. [3DOH50K](https://www.yangangwang.com/papers/ZHANG-OOH-2020-03.html) (CVPR'2020)
23. [Penn Action](http://dreamdragon.github.io/PennAction/) (ICCV'2012)
24. [3D-People](https://cv.iri.upc-csic.es/) (ICCV'2019)
25. [PoseTrack18](https://posetrack.net/users/download.php) (CVPR'2018)
26. [PROX](https://prox.is.tue.mpg.de/) (ICCV'2019)
27. [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) (ECCV'2018)
28. [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) (CVPR'2017)
29. [UP-3D](https://files.is.tuebingen.mpg.de/classner/up/) (CVPR'2017)
30. [VLOG](https://github.com/akanazawa/human_dynamics/blob/master/doc/vlog_people.md) (CVPR'2019)
31. [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) (CVPR'2019)

</details>


  Please refer to [datasets.md](./configs/datasets/README.md) for training configs and results.

- **Benchmarks on different dataset combinations**

### Mixed-datasets

<details open>
<summary>(click to collapse)</summary>

1. Mix 1: H36M, MI, COCO
2. Mix 2: H36M, MI, EFT-COCO
3. Mix 3: H36M, MI, EFT-COCO, MPII
4. Mix 4: H36M, MuCo, EFT-COCO
5. Mix 5: H36M, MI, COCO, LSP, LSPET, MPII
6. Mix 6: EFT-[COCO, MPII, LSPET], SPIN-MI, H36M
7. Mix 7: EFT-[COCO, MPII, LSPET], MuCo, H36M, PROX
8. Mix 8: EFT-[COCO, PT, LSPET], MI, H36M
9. Mix 9: EFT-[COCO, PT, LSPET, OCH], MI, H36M
10. Mix 10: PROX, MuCo, EFT-[COCO, PT, LSPET, OCH], UP-3D, MTP, Crowdpose
11. Mix 11: EFT-[COCO, MPII, LSPET], MuCo, H36M

</details>

  Please refer to [mixed-datasets.md](./configs/mixed/README.md) for training configs and results.

### Backbones

<details open>
<summary>(click to collapse)</summary>

- [x] ResNet-50, -101, -152 (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] HRNet (CVPR'2019)
- [x] EfficientNet
- [x] ViT
- [x] Swin
- [x] Twins

</details>

  <!-- Train with a suite of augmentation techniques for a more robust model -->
  Please refer to [backbone.md](./configs/backbones/README.md) for training configs and results.

### Backbone-initialisation

We find that transfering knowledge from a pose estimation model gives more competitive performance.

Initialised backbones:
<details open>
<summary>(click to collapse)</summary>

1. ResNet-50 ImageNet (default)
2. ResNet-50 MPII
3. ResNet-50 COCO
4. HRNet-W32 ImageNet
5. HRNet-W32 MPII
6. HRNet-W32 COCO
7. Twins-SVT ImageNet
8. Twins-SVT MPII
9. Twins-SVT COCO

</details>

  <!-- Train with a suite of augmentation techniques for a more robust model -->
  Please refer to [backbone.md](./configs/backbones/README.md) for training configs and results.


### Augmentations

New augmentations:
<details open>
<summary>(click to collapse)</summary>

1. Coarse dropout
2. Grid dropout
3. Photometric distortion
4. Random crop
5. Hard erasing
6. Soft erasing
7. Self-mixing
8. Synthetic occlusion
9. Synthetic occlusion over keypoints

</details>


  <!-- Train with a suite of augmentation techniques for a more robust model -->
  Please refer to [augmentation.md](./configs/pretrained-backbones/README.md) for training configs and results.

### Losses

We find that training with L1 loss gives more competitive performance. Please refer to [mixed-datasets-l1.md](./configs/mixed-l1/README.md) for training configs and results.

### Downloads

We provide trained models from the optimal configurations for download and inference. Please refer to [combine.md](./configs/combine/README.md) for training configs and results.


| Dataset   | Backbone | 3DPW (PA-MPJPE)    | Download |
|:------:|:-------:|:------:|:-------:|
| H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | 51.66 | [model](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| H36M, MI, COCO, LSP, LSPET, MPII | HRNet-W32 | 49.18 | [model](https://drive.google.com/file/d/1GV7T8ub5CCw_Tt0e-6SYlI_vimEl_ETy/view?usp=sharing) |
| H36M, MI, COCO, LSP, LSPET, MPII | Twins-SVT | 48.77 | [model](https://drive.google.com/file/d/1UOLovoUUCvwXE14yoaJO9o-vpaeSvMPA/view?usp=sharing) |
| H36M, MI, COCO, LSP, LSPET, MPII | Twins-SVT | 47.70 | [model](https://drive.google.com/file/d/1zk2JanLjkJ1W0TIAPhUaSZtVB-uayWFi/view?usp=sharing) |
| EFT-[COCO, LSPET, MPII], H36M, SPIN-MI | HRNet-W32 | 47.68 | [model](https://drive.google.com/file/d/1NkijOkAKeNaDUx5XsF8nhL-MiboIcLRu/view?usp=sharing) |
| EFT-[COCO, LSPET, MPII], H36M, SPIN-MI  | Twins-SVT | 47.31 | [model](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| H36M, MI, EFT-COCO | HRNet-W32 | 48.08 | [model](https://drive.google.com/file/d/19poA9gmmuOlMbcREBGRF70EqyM00bDxi/view?usp=sharing) |
| H36M, MI, EFT-COCO  | Twins-SVT | 48.27 | [model](https://drive.google.com/file/d/1hnk8cMQ2QbA1jrZyHaRqqAN1jXdBo7ed/view?usp=sharing) |
| H36M, MuCo, EFT-COCO  | Twins-SVT | 47.92 | [model](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |


### Algorithms

We benchmarked our major findings on several algorithms and hope to add more in the future. Please refer to [algorithms.md](./configs/algorithms/README.md) for training configs and logs.
<details open>
<summary>(click to collapse)</summary>

1. SPIN
2. GraphCMR
3. PARE
4. Mesh Graphormer

</details>

<!--
- **Evaluation for different benchmarks**

<!-- Easily obtain benchmarks on their trained model on five test sets (1) 3DPW-test (2) H36M (P1/ P2) test (3) EFT-OCHuman-test (4)  EFT-COCO-Val (5) EFT-LSPET-test -->
<!-- Test sets for evaluation:
<details open>
<summary>(click to collapse)</summary>

- 3DPW-test (P2)
- H36m-test (P2)
- EFT-COCO-val
- EFT-LSPET-test
- EFT-OCHuman-test

</details>


- **FLOPs and Param evaluation for trained model**

  Evaluate flops and params for trained model -->
<!--
- **Provide training log and model pths for inference**

  Will add links to Gdrive -->


## Installation

General set-up instructions follow that of [MMHuman3d](https://openmmlab.com/mmhuman3d). Please refer to [install.md](./install.md) for installation.

<!-- ## Data Preparation

Please refer to [data_preparation.md](./preprocess_dataset.md) for data preparation.

## Body Model Preparation

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
  - Neutral model can be downloaded from [SMPLify](https://smplify.is.tue.mpg.de/).
  - All body models have to be renamed in `SMPL_{GENDER}.pkl` format. <br/>
    For example, `mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl SMPL_NEUTRAL.pkl`
- [J_regressor_extra.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_extra.npy?versionId=CAEQHhiBgIDD6c3V6xciIGIwZDEzYWI5NTBlOTRkODU4OTE1M2Y4YTI0NTVlZGM1)
- [J_regressor_h36m.npy](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/J_regressor_h36m.npy?versionId=CAEQHhiBgIDE6c3V6xciIDdjYzE3MzQ4MmU4MzQyNmRiZDA5YTg2YTI5YWFkNjRi)
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)

Download the above resources and arrange them in the following file structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── body_models
        ├── J_regressor_extra.npy
        ├── J_regressor_h36m.npy
        ├── smpl_mean_params.npz
        └── smpl
            ├── SMPL_FEMALE.pkl
            ├── SMPL_MALE.pkl
            └── SMPL_NEUTRAL.pkl
``` -->

## Train

### Training with a single / multiple GPUs

```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --no-validate
```
Example: using 1 GPU to train HMR.
```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --gpus 1 --no-validate
```

### Training with Slurm

If you can run MMHuman3D on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --no-validate
```

Common optional arguments include:
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--no-validate`: Whether not to evaluate the checkpoint during training.

Example: using 8 GPUs to train HMR on a slurm cluster.
```shell
./tools/slurm_train.sh my_partition my_job configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr 8 --no-validate
```

You can check [slurm_train.sh](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/slurm_train.sh) for full arguments and environment variables.


## Evaluation

There's five benchmarks for evaluation:
- 3DPW-test (P2)
- H36m-test (P2)
- EFT-COCO-val
- EFT-LSPET-test
- EFT-OCHuman-test

### Evaluate with a single GPU / multiple GPUs

```shell
python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${CHECKPOINT} --metrics=${METRICS}
```
Example:
```shell
python tools/test.py configs/hmr/resnet50_hmr_pw3d.py --work-dir=work_dirs/hmr work_dirs/hmr/latest.pth --metrics pa-mpjpe mpjpe
```

### Evaluate with slurm

If you can run MMHuman3D on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`.

```shell
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} ${CHECKPOINT} --metrics ${METRICS}
```
Example:
```shell
./tools/slurm_test.sh my_partition test_hmr configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr work_dirs/hmr/latest.pth 8 --metrics pa-mpjpe mpjpe
```


## FLOPs

`tools/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) and [MMDetection](https://github.com/open-mmlab/mmdetection) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the results like this.

```text
==============================
Input shape: (3, 1280, 800)
Flops: 239.32 GFLOPs
Params: 37.74 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the
 number is absolutely correct. You may well use the result for simple
  comparisons, but double check it before you adopt it in technical reports or papers.

1. FLOPs are related to the input shape while parameters are not. The default
 input shape is (1, 3, 224, 224).
2. Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.


## Citation
If you find our work useful for your research, please consider citing the paper:
```
@article{
  title={Benchmarking and Analyzing 3D Human Pose and Shape Estimation Beyond Algorithms},
  author={Pang, Hui En and Cai, Zhongang and Yang, Lei and Zhang, Tianwei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2209.10529},
  year={2022}
}
```

## License

Distributed under the S-Lab License. See `LICENSE` for more information.

## Acknowledgements

This study is supported by NTU NAP, MOE AcRF Tier 2 (T2EP20221-0033), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).
