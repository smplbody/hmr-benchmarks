# Combine

## Introduction

We provide the config files for training and pretrained models for inference on the optimal configurations.

## Notes

Download the pretrained backbones from here:
1. [resnet50_coco_pose.pth](https://drive.google.com/file/d/1K1K1AUxL20Grq8rsyLJ6OdZE0oXY_CNY/view?usp=sharing)
2. [hrnet_coco_pose.pth](https://drive.google.com/file/d/1Dt1eRN_YnltaDBBe0JU8f6oSfhB2pxeh/view?usp=sharing)
3. [twins_svt_coco_pose.pth](https://drive.google.com/file/d/1Fcq_4G3ccM-xpmBK4M--Lu3xCXFrQ_ui/view?usp=sharing)



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
        ├── resnet50_coco_pose.pth
        ├── hrnet_coco_pose.pth
        └── twins_svt_coco_pose.pth

```

## Results and Models

We evaluate HMR on 3DPW. Values are PA-MPJPE.

| Config | Dataset   | Backbone | 3DPW    | Download |
|:------:|:-------:|:------:|:-------:|:------:|
| [resnet50_hmr_mix1_coco_l1.py](resnet50_hmr_mix1_coco_l1.py) | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | 51.66 | [model](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| [hrnet_hmr_mix1_coco_l1.py](hrnet_hmr_mix1_coco_l1.py) | H36M, MI, COCO, LSP, LSPET, MPII | HRNet-W32 | 49.18 | [model](https://drive.google.com/file/d/1GV7T8ub5CCw_Tt0e-6SYlI_vimEl_ETy/view?usp=sharing) |
| [twins_svt_hmr_mix1_coco_l1.py](twins_svt_hmr_mix1_coco_l1.py) | H36M, MI, COCO, LSP, LSPET, MPII | Twins-SVT | 48.77 | [model](https://drive.google.com/file/d/1UOLovoUUCvwXE14yoaJO9o-vpaeSvMPA/view?usp=sharing) |
| [twins_svt_hmr_mix1_coco_l1_aug.py](twins_svt_hmr_mix1_coco_l1_aug.py) | H36M, MI, COCO, LSP, LSPET, MPII | Twins-SVT | 47.70 | [model](https://drive.google.com/file/d/1zk2JanLjkJ1W0TIAPhUaSZtVB-uayWFi/view?usp=sharing) |
| [hrnet_hmr_mix4_coco_l1_aug.py](hrnet_hmr_mix4_coco_l1_aug.py) | EFT-[COCO, LSPET, MPII], H36M, SPIN-MI | HRNet-W32 | 47.68 | [model](https://drive.google.com/file/d/1NkijOkAKeNaDUx5XsF8nhL-MiboIcLRu/view?usp=sharing) |
| [twins_svt_hmr_mix4_coco_l1.py](twins_svt_hmr_mix4_coco_l1.py) | EFT-[COCO, LSPET, MPII], H36M, SPIN-MI  | Twins-SVT | 47.31 | [model](https://drive.google.com/file/d/1ostUnbf8MIVerlLo0AAP7As4V_gs-41k/view?usp=share_link) |
| [hrnet_hmr_mix2_coco_l1_aug.py](hrnet_hmr_mix2_coco_l1_aug.py) | H36M, MI, EFT-COCO | HRNet-W32 | 48.08 | [model](https://drive.google.com/file/d/19poA9gmmuOlMbcREBGRF70EqyM00bDxi/view?usp=sharing) |
| [twins_svt_hmr_mix2_coco_l1.py](twins_svt_hmr_mix2_coco_l1.py) | H36M, MI, EFT-COCO  | Twins-SVT | 48.27 | [model](https://drive.google.com/file/d/1hnk8cMQ2QbA1jrZyHaRqqAN1jXdBo7ed/view?usp=sharing) |
| [twins_svt_hmr_mix6_coco_l1.py](twins_svt_hmr_mix6_coco_l1.py) | H36M, MuCo, EFT-COCO  | Twins-SVT | 47.92 | [model](https://drive.google.com/file/d/1ZQG0LCArhM3k-C1IQ-ZEc3cKAUbH8xFf/view?usp=share_link) |
