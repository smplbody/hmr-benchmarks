# Mixed datasets

## Introduction

We provide the config files for training on different losses for different dataset combination:
1. Mix 1: H36M, MI, COCO
2. Mix 2: H36M, MI, EFT-COCO
3. Mix 5: H36M, MI, COCO, LSP, LSPET, MPII
4. Mix 6: EFT-[COCO, MPII, LSPET], SPIN-MI, H36M
5. Mix 7: EFT-[COCO, MPII, LSPET], MuCo, H36M, PROX
6. Mix 8: EFT-[COCO, PT, LSPET], MI, H36M
7. Mix 11: EFT-[COCO, MPII, LSPET], MuCo, H36M


## Results and Models

We evaluate trained models on 3DPW. Values are MPJPE/PA-MPJPE.

| Mixes | Datasets  Config | 3DPW (w/ L1)   | 3DPW (w/o L1)   |
|:------:|:------:|:-------:|:-------:|:-------:|
| Mix 1 | H36M, MI, COCO | [resnet50_hmr_mix1.py](resnet50_hmr_mix1.py) | 57.01 | 66.14 |
| Mix 2 | H36M, MI, EFT-COCO |[resnet50_hmr_mix2.py](resnet50_hmr_mix2.py) | 55.25 | 55.98 |
| Mix 5 | H36M, MI, COCO, LSP, LSPET, MPII |[resnet50_hmr_mix5.py](resnet50_hmr_mix5.py) | 58.20 | 64.55 |
| Mix 6 | EFT-[COCO, MPII, LSPET], SPIN-MI, H36M |[resnet50_hmr_mix6.py](resnet50_hmr_mix6.py) | 53.62 | 55.47 |
| Mix 7 | EFT-[COCO, MPII, LSPET], MuCo, H36M, PROX |[resnet50_hmr_mix7.py](resnet50_hmr_mix7.py) | 52.93 | 53.44 |
| Mix 8 | EFT-[COCO, PT, LSPET], MI, H36M |[resnet50_hmr_mix8.py](resnet50_hmr_mix8.py) | 53.43 | 55.97 |
| Mix 11 | EFT-[COCO, MPII, LSPET], MuCo, H36M |[resnet50_hmr_mix11.py](resnet50_hmr_mix11.py) | 53.17 | 52.54 |
