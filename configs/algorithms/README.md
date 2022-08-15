# Algorithms

## Introduction

We provide the config files for training and pretrained models for inference on the optimal configurations.

## Notes

Download the pretrained backbones from here:
1. [resnet50_coco_pose.pth](https://drive.google.com/file/d/1K1K1AUxL20Grq8rsyLJ6OdZE0oXY_CNY/view?usp=sharing)
2. [hrnet_coco_pose.pth](https://drive.google.com/file/d/1Dt1eRN_YnltaDBBe0JU8f6oSfhB2pxeh/view?usp=sharing)
3. [hrnetw48_coco_pose.pth](https://drive.google.com/file/d/1Fcq_4G3ccM-xpmBK4M--Lu3xCXFrQ_ui/view?usp=sharing)



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
        └── hrnetw48_coco_pose.pth

```

## Results and Models

We evaluate on 3DPW. Values are PA-MPJPE.

Our baseline models for HMR, SPIN and GraphCMR can reach the reported results in the respective works. For PARE, the original work trains the model on MPII for pose estimation task and later on EFT-COCO for mesh recovery before training on he full set of datasets. To keep consistent with the practice adopted throughout our work, we benchmark PARE by training it fro scratch with only ImageNet initialisation. For Graphormer, the original work evaluates on H36M every epoch before fine-tuning the best H36M model on 3DPW-train (Protocol 1) for 5 epochs. To keep consistent, we adopt Protocol 2 instead. 


| Algorithms | Datasets   | Backbone | Initialisation | Normal | L1 | L1+COCO | L1+COCO+Aug |
|:------:|:-------:|:------:|:-------:|:------:|:------:|:-------:|:------:|
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | ImageNet | 64.55 | 58.20 | 51.8 | 51.66 |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | HMR(ImageNet) | 59.2 | 57.08 | 51.54 | 50.69 |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | ImageNet | 70.51 | 67.2 | 61.74 | 60.26 |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | ImageNet | 61.99 | 61.13 | 59.98 | 51.66 |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48 | ImageNet | 63.18 | 63.47 | 59.66 | 58.82 |

| Config | Datasets   | Backbone |  Variant | 3DPW    | Log |
|:------:|:-------:|:------:|:-------:|:------:|:------:|
<!-- | [resnet50_hmr_mix1.py](resnet50_hmr_mix1_coco_l1.py) | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 64.55 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) | -->
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 64.55v| [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 | 58.20 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 + COCO | 51.8 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 + COCO + Aug | 51.66 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 59.2 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 57.08 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 51.54 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 50.69 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | Normal | 70.51 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D  ResNet-50 | Normal | 67.2 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | Normal | 61.74 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | Normal | 60.26 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | Normal | 61.99 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | Normal | 61.13 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | Normal | 59.98 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | Normal | 51.66 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | Normal | 63.18 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | Normal | 63.47 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | Normal | 59.66 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | Normal | 58.82 | [log](https://drive.google.com/file/d/1ifPYeQY8w-uJzl6yFejaTy_O86OmrjNH/view?usp=sharing) |
