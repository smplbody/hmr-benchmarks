# Augmentations

## Introduction

We provide the config files for training on different augmentations on H36M and EFT-COCO:
1. Coarse dropout
2. Grid dropout
3. Photometric distortion
4. Random crop
5. Hard erasing
6. Soft erasing
7. Self-mixing
8. Synthetic occlusion
9. Synthetic occlusion over keypoints



## Results and Models

We evaluate trained models on 3DPW. Values are MPJPE/PA-MPJPE.

On H36M

| Config | 3DPW    |
|:------:|:-------:|
| [resnet50_hmr_h36m.py](resnet50_hmr_h36m.py) | 105.42 / 60.11 |
| [resnet50_hmr_coarse_dropout_h36m.py](resnet50_hmr_coarse_dropout_h36m.py) | 112.46 / 64.55 |
| [resnet50_hmr_grid_dropout_h36m.py](resnet50_hmr_grid_dropout_h36m.py) | 112.67 / 63.36 |
| [resnet50_hmr_photometric_distortion_h36m.py](resnet50_hmr_photometric_distortion_h36m.py) | 107.13 / 62.13 |
| [resnet50_hmr_rand_crop_h36m.py](resnet50_hmr_rand_crop_h36m.py) | 114.43 / 64.95 |
| [resnet50_hmr_rand_occ_h36m.py](resnet50_hmr_rand_occ_h36m.py) | 112.34 / 67.53 |
| [resnet50_hmr_soft_erase_h36m.py](resnet50_hmr_soft_erase_h36m.py) | 118.15 / 65.16 |
| [resnet50_hmr_self_mix_h36m.py](resnet50_hmr_self_mix_h36m.py) | 111.46 / 62.81 |
| [resnet50_hmr_syn_occ_h36m.py](resnet50_hmr_syn_occ_h36m.py) | 110.42 / 62.78 |
| [resnet50_hmr_syn_occkp_h36m.py](resnet50_hmr_syn_occkp_h36m.py) | 100.75 / 59.13 |


On EFT-COCO

| Config | 3DPW    |
|:------:|:-------:|
| [resnet50_hmr_eftcoco.py](resnet50_hmr_eftcoco.py) | 105.42 / 60.11 |
| [resnet50_hmr_coarse_dropout_eftcoco.py](resnet50_hmr_coarse_dropout_eftcoco.py) | 112.46 / 64.55 |
| [resnet50_hmr_grid_dropout_eftcoco.py](resnet50_hmr_grid_dropout_eftcoco.py) | 112.67 / 63.36 |
| [resnet50_hmr_photometric_distortion_eftcoco.py](resnet50_hmr_photometric_distortion_eftcoco.py) | 107.13 / 62.13 |
| [resnet50_hmr_rand_crop_eftcoco.py](resnet50_hmr_rand_crop_eftcoco.py) | 114.43 / 64.95 |
| [resnet50_hmr_rand_occ_eftcoco.py](resnet50_hmr_rand_occ_eftcoco.py) | 112.34 / 67.53 |
| [resnet50_hmr_soft_erase_eftcoco.py](resnet50_hmr_soft_erase_eftcoco.py) | 118.15 / 65.16 |
| [resnet50_hmr_self_mix_eftcoco.py](resnet50_hmr_self_mix_eftcoco.py) | 111.46 / 62.81 |
| [resnet50_hmr_syn_occ_eftcoco.py](resnet50_hmr_syn_occ_eftcoco.py) | 110.42 / 62.78 |
| [resnet50_hmr_syn_occkp_eftcoco.py](resnet50_hmr_syn_occkp_eftcoco.py) | 100.75 / 59.13 |