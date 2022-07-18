<div align="center">

# Benchmarking 3D Pose and Shape Estimation Beyond Algorithms


### [Experiments](#experiments) |[Installation](#installation) | [Train](#train) | [Evaluation](#evaluation) | [FLOPs](#flops) |



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
    <img src="resources/dance001.gif" width="70%">
</p>

## Experiments

We have added multiple major features on top of MMHuman3D.


<!-- - **Provide annotation files**

  Will add links to Gdrive. Future works can use these files to train on any of the 31 dataset. -->

<!-- - **Provide config files for 31 single and multi-dataset benchmarks** -->
- **Benchmarks on 31 datasets**
<!--
  Future works can use these configs for training and obtain benchmarks on HMR for baseline comparison on their selected dataset mixes and partition. -->
Supported datasets:
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

  Please refer to [datasets.md](./configs/datasets/README.md) for training configs and results.

- **Benchmarks on different dataset combinations**

Dataset mixes:
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

  Please refer to [mixed-datasets.md](./configs/mixed/README.md) for training configs and results.

- **Benchmarks on different backbones**

Supported backbones:
- [x] ResNet-50, -101, -152 (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] HRNet (CVPR'2019)
- [x] EfficientNet
- [x] ViT
- [x] Swin
- [x] Twins

  <!-- Train with a suite of augmentation techniques for a more robust model -->
  Please refer to [backbone.md](./configs/backbones/README.md) for training configs and results.

- **Benchmarks on different backbone initialisation**

Supported backbones:
- [x] ResNet-50, -101, -152 (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] HRNet (CVPR'2019)
- [x] EfficientNet
- [x] ViT
- [x] Swin
- [x] Twins

  <!-- Train with a suite of augmentation techniques for a more robust model -->
  Please refer to [backbone.md](./configs/backbones/README.md) for training configs and results.


- **Benchmarks on different augmentation techniques**

We find that transfering knowledge from a pose estimation model gives more competitive performance.

Initialised backbones:
1. ResNet-50 ImageNet (default)
2. ResNet-50 MPII
3. ResNet-50 COCO
4. HRNet-W32 ImageNet
5. HRNet-W32 MPII
6. HRNet-W32 COCO
7. Twins-SVT ImageNet
8. Twins-SVT MPII
9. Twins-SVT COCO

  <!-- Train with a suite of augmentation techniques for a more robust model -->
  Please refer to [augmentation.md](./configs/pretrained-backbones/README.md) for training configs and results.

- **Benchmarks on different losses**

We find that training with L1 loss gives more competitive performance. Please refer to [mixed-datasets-l1.md](./configs/mixed-l1/README.md) for training configs and results.

- **Provide trained models for optimal configurations**

We find that training with L1 loss gives more competitive performance. Please refer to [mixed-datasets-l1.md](./configs/mixed-l1/README.md) for training configs and results.


- **Evaluation for different benchmarks**

<!-- Easily obtain benchmarks on their trained model on five test sets (1) 3DPW-test (2) H36M (P1/ P2) test (3) EFT-OCHuman-test (4)  EFT-COCO-Val (5) EFT-LSPET-test -->
Test sets for evaluation:
- 3DPW-test (P2)
- H36m-test (P2)
- EFT-COCO-val
- EFT-LSPET-test
- EFT-OCHuman-test

- **FLOPs and Param evaluation for trained model**

  Evaluate flops and params for trained model
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
