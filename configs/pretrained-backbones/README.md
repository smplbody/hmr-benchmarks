# Pretrained backbones

## Introduction

Download the pretrained backbones from here:
1. [resnet50_mpii_pose.pth](https://drive.google.com/file/d/1XEyYR88S9LnAh_bztybtWhBix6vQgg5k/view?usp=sharing)
2. [resnet50_coco_pose.pth](https://drive.google.com/file/d/1K1K1AUxL20Grq8rsyLJ6OdZE0oXY_CNY/view?usp=sharing)
3. [hrnet_imagenet.pth](https://drive.google.com/file/d/1snrLDyHgpTXximcJX6EqX7M8okQHVjH7/view?usp=sharing)
4. [hrnet_mpii_pose.pth](https://drive.google.com/file/d/1JaKYRbP-hKKZCwAqlvDQ5hIhueBYGf0i/view?usp=sharing)
5. [hrnet_coco_pose.pth](https://drive.google.com/file/d/1Dt1eRN_YnltaDBBe0JU8f6oSfhB2pxeh/view?usp=sharing)
3. [twins_svt_imagenet.pth](https://drive.google.com/file/d/155neTTkGZ_jtNbRS-OYwJWwsbx59jtzK/view?usp=sharing)
4. [twins_svt_mpii_pose.pth](https://drive.google.com/file/d/1RItsH4dDQmsk6Xc9wyaKyW3wYWFKpN6p/view?usp=sharing)
5. [twins_svt_coco_pose.pth](https://drive.google.com/file/d/1Fcq_4G3ccM-xpmBK4M--Lu3xCXFrQ_ui/view?usp=sharing)



Download the above resources and arrange them in the following file structure:

```text
mmhuman3d
├── mmhuman3d
├── docs
├── tests
├── tools
├── configs
└── data
    └── checkpoints
        ├── resnet50_mpii_pose.pth
        ├── resnet50_coco_pose.pth
        ├── hrnet_imagenet.pth
        ├── hrnet_mpii_pose.pth
        ├── hrnet_coco_pose.pth
        ├── twins_svt_imagenet.pth
        ├── twins_svt_mpii_pose.pth
        └── twins_svt_coco_pose.pth

```

## Results and Models

We evaluate trained models on 3DPW. Values are MPJPE/PA-MPJPE.

| Backbones | Weights | Config | 3DPW    |
|:------:|:------:|:-------:|:-------:|
|ResNet-50| ImageNet | [resnet50_hmr_imagenet.py](resnet50_hmr_imagenet.py) | 64.55 |
|ResNet-50| MPII | [resnet50_hmr_mpii.py](resnet50_hmr_pw3d_mpii.py) | 60.60 |
|ResNet-50| COCO | [resnet50_hmr_coco.py](resnet50_hmr_coco.py) | 57.26 |
|HRNet-W32| ImageNet | [hrnet_hmr_imagenet.py](hrnet_hmr_imagenet.py) | 64.27 |
|HRNet-W32| MPII | [hrnet_hmr_mpii.py](hrnet_hmr_mpii.py) | 55.93 |
|HRNet-W32| COCO | [hrnet_hmr_coco.py](hrnet_hmr_coco.py) | 54.47 |
|Twins-SVT| ImageNet | [twins_svt_hmr_imagenet.py](twins_svt_hmr_imagenet.py) | 60.11 |
|Twins-SVT| MPII | [twins_svt_hmr_mpii.py](twins_svt_hmr_mpii.py) | 56.80 |
|Twins-SVT| COCO | [twins_svt_hmr_coco.py](twins_svt_hmr_coco.py) | 52.61 |