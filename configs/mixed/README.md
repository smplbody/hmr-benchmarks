# Mixed datasets

## Introduction

We provide the config files for training on different dataset combination:
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



## Results and Models

We evaluate trained models on 3DPW. Values are MPJPE/PA-MPJPE.

| Mixes | Datasets  Config | 3DPW    |
|:------:|:------:|:-------:|:-------:|
| Mix 1 | H36M, MI, COCO | [resnet50_hmr_mix1.py](resnet50_hmr_mix1.py) | 66.14 |
| Mix 2 | H36M, MI, EFT-COCO |[resnet50_hmr_mix2.py](resnet50_hmr_mix2.py) | 55.98 |
| Mix 3 | H36M, MI, EFT-COCO, MPII |[resnet50_hmr_mix3.py](resnet50_hmr_mix3.py) | 56.12 |
| Mix 4 | H36M, MuCo, EFT-COCO |[resnet50_hmr_mix4.py](resnet50_hmr_mix4.py) | 53.90 |
| Mix 5 | H36M, MI, COCO, LSP, LSPET, MPII |[resnet50_hmr_mix5.py](resnet50_hmr_mix5.py) | 64.55 |
| Mix 6 | EFT-[COCO, MPII, LSPET], SPIN-MI, H36M |[resnet50_hmr_mix6.py](resnet50_hmr_mix6.py) | 55.47 |
| Mix 7 | EFT-[COCO, MPII, LSPET], MuCo, H36M, PROX |[resnet50_hmr_mix7.py](resnet50_hmr_mix7.py) | 53.44 |
| Mix 8 | EFT-[COCO, PT, LSPET], MI, H36M |[resnet50_hmr_mix8.py](resnet50_hmr_mix8.py) | 55.97 |
| Mix 9 | EFT-[COCO, PT, LSPET, OCH], MI, H36M |[resnet50_hmr_mix9.py](resnet50_hmr_mix9.py) | 55.59 |
| Mix 10 | PROX, MuCo, EFT-[COCO, PT, LSPET, OCH], UP-3D, MTP, Crowdpose |[resnet50_hmr_mix10.py](resnet50_hmr_mix10.py) | 57.84 |
| Mix 11 | EFT-[COCO, MPII, LSPET], MuCo, H36M |[resnet50_hmr_mix11.py](resnet50_hmr_mix11.py) | 52.54 |