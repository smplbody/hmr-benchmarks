# Algorithms

## Introduction

We provide the config files for training and pretrained models for inference on the optimal configurations.

## Notes

Download the pretrained backbones from here:
1. [resnet50_coco_pose.pth](https://drive.google.com/file/d/1K1K1AUxL20Grq8rsyLJ6OdZE0oXY_CNY/view?usp=sharing)
2. [hrnet_coco_pose.pth](https://drive.google.com/file/d/1Dt1eRN_YnltaDBBe0JU8f6oSfhB2pxeh/view?usp=sharing)
3. [hrnetw48_coco_pose.pth](https://drive.google.com/file/d/1Viiqq-2t-KT1DJvREvsShc-UkN6RwdSp/view?usp=sharing)



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
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | ImageNet | 64.55 | 58.20 | 51.80 | 51.66 |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | HMR(ImageNet) | 59.00 | 57.08 | 51.54 | 50.69 |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | ImageNet | 70.51 | 67.20 | 61.74 | 60.26 |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | ImageNet | 61.99 | 61.13 | 59.98 | 58.32 |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48 | ImageNet | 63.18 | 63.47 | 59.66 | 58.82 |

| Config | Datasets   | Backbone |  Variant | 3DPW    | Log |
|:------:|:-------:|:------:|:-------:|:------:|:------:|
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 64.55| [log](https://drive.google.com/file/d/1AG7XZltGzx1dEQZvk59mx3fL-SWsNPyR/view?usp=sharing) |
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 | 58.20 | [log](https://drive.google.com/file/d/1rnsOXVL7rBx10NV3nt0tim5OHYav1iUi/view?usp=sharing) |
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 + COCO | 51.8 | [log](https://drive.google.com/file/d/1Bo1kSU6WEE3nJxOvVwmwuQJi4bigzv22/view?usp=sharing) |
| HMR | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 + COCO + Aug | 51.66 | [log](https://drive.google.com/file/d/1Bo1kSU6WEE3nJxOvVwmwuQJi4bigzv22/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | Normal | 59.00 | [log](https://drive.google.com/file/d/1mTo33VcB7N0yEKM-sWQRmRi0SFXpvxVM/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 | 57.08 | [log](https://drive.google.com/file/d/1OecAPxOKGqylELkcRx0TAbcsWgd57Zpm/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 + COCO | 51.54 | [log](https://drive.google.com/file/d/13su1hD6qEgsIlsLicLul0i9imSenuWER/view?usp=sharing) |
| SPIN | H36M, MI, COCO, LSP, LSPET, MPII | ResNet-50 | L1 + COCO + Aug | 50.69 | [log](https://drive.google.com/file/d/1vY3_XGaa7p19ttYiSoPtYPSs0waMQnOp/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | Normal | 70.51 | [log](https://drive.google.com/file/d/1TSpMR4zlGWkksopNNIN20v7Y0YDg4lp1/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | L1 | 67.2 | [log](https://drive.google.com/file/d/1DrmUF4DK_-G-3wFiWteo16hvwpmYG_Ow/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | L1 + COCO | 61.74 | [log](https://drive.google.com/file/d/1DrmUF4DK_-G-3wFiWteo16hvwpmYG_Ow/view?usp=sharing) |
| GraphCMR | H36M, COCO, LSP, LSPET, MPII, UP3D | ResNet-50 | L1 + COCO + Aug | 60.26 | [log](https://drive.google.com/file/d/1P6gvswzZ2VNfPP29J1QIl3WVhaGT4ASB/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | Normal | 61.99 | [log](https://drive.google.com/file/d/1MEG7FjIeGc_gXNnPBL0sqhptH13S10zH/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | L1 | 61.13 | [log](https://drive.google.com/file/d/1Zd11y2IOhvBMxoS0fFOgOUM2JjDwDYRE/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | L1 + COCO | 59.98 | [log](https://drive.google.com/file/d/1cDtmOFa8l1HAzJIjgoXlJrjAcdVMTlze/view?usp=sharing) |
| PARE | H36M, MI, EFT-[COCO, LSPET, MPII] | HRNet-W32 | L1 + COCO + Aug | 58.32 | [log](https://drive.google.com/file/d/15cezt0ZZllaP-uzI1ZGh-U9PeN_AwIyb/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | Normal | 63.18 | [log](https://drive.google.com/file/d/1wRyd5fEn07QP7BonMnFOIT4nk7gWon40/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | L1 | 63.47 | [log](https://drive.google.com/file/d/1bjYl4JIE1l2KAo-HDo2cZjPxFoOvJjgk/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | L1 + COCO | 59.66 | [log](https://drive.google.com/file/d/1WWqr9ruRHP-3I-81dcdCWS57tLvtUeoh/view?usp=sharing) |
| Graphormer | H36M, MuCo, COCO, UP3D, MPII | HRNet-W48  | L1 + COCO + Aug | 58.82 | [log](https://drive.google.com/file/d/10_smiIdZUip1CBH6Mr66qVifxvPIxIgC/view?usp=sharing) |
