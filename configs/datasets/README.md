# Augmentations

## Introduction

We provide the config files for training on 31 different datasets:
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



## Results and Models

We evaluate trained models on 3DPW. Values are PA-MPJPE.

| Datasets | Config | 3DPW    |
|:------:|:------:|:-------:|
| AGORA | [resnet50_hmr_agora.py](resnet50_hmr_agora.py) | 77.94 |
| AI Challenger | [resnet50_hmr_aic.py](resnet50_hmr_aic.py) | 111.66 |
| COCO | [resnet50_hmr_coco.py](resnet50_hmr_coco.py) | 93.18 |
| COCO-Wholebody | [resnet50_hmr_coco_wholebody.py](resnet50_hmr_coco_wholebody.py) | 85.27 |
| Crowdpose | [resnet50_hmr_crowdpose.py](resnet50_hmr_crowdpose.py) | 99.97  |
| EFT-COCO-Part | [resnet50_hmr_eft_coco_part.py](resnet50_hmr_eft_coco_part.py) |67.81  |
| EFT-COCO | [resnet50_hmr_eft_coco.py](resnet50_hmr_eft_coco.py) |  60.82  |
| EFT-LSPET | [resnet50_hmr_eft_lspet.py](resnet50_hmr_eft_lspet.py) |100.53  |
| EFT-OCHuman | [resnet50_hmr_eft_ochuman.py](resnet50_hmr_eft_ochuman.py) | 94.01 |
| EFT-PoseTrack | [resnet50_hmr_eft_posetrack.py](resnet50_hmr_eft_posetrack.py) | 75.17 |
| EFT-MPII | [resnet50_hmr_eft_mpii.py](resnet50_hmr_eft_mpii.py) | 77.67 |
| H36M | [resnet50_hmr_h36m.py](resnet50_hmr_h36m.py) | 124.55 |
| InstaVariety | [resnet50_hmr_instavariety.py](resnet50_hmr_instavariety.py) | 88.93 |
| LIP | [resnet50_hmr_lip.py](resnet50_hmr_lip.py) | 96.47 |
| LSP | [resnet50_hmr_lsp.py](resnet50_hmr_lsp.py) | 111.45 |
| LSP-Extended | [resnet50_hmr_lspet.py](resnet50_hmr_lspet.py) | 112.26  |
| MPI-INF-3DHP | [resnet50_hmr_mpi_inf_3dhp.py](resnet50_hmr_mpi_inf_3dhp.py) | 107.15  |
| MPII | [resnet50_hmr_mpii.py](resnet50_hmr_mpii.py) | 98.18 |
| MTP | [resnet50_hmr_mtp.py](resnet50_hmr_mtp.py) | 87.03 |
| MuCo-3DHP | [resnet50_hmr_muco.py](resnet50_hmr_muco.py) | 78.05  |
| MuPoTs-3D | [resnet50_hmr_mupots3d.py](resnet50_hmr_mupots3d.py) | 95.83 |
| OCHuman | [resnet50_hmr_ochuman.py](resnet50_hmr_ochuman.py) | 130.55  |
| 3DOH50K | [resnet50_hmr_oh50k3d.py](resnet50_hmr_oh50k3d.py) | 114.48 |
| Penn-Action | [resnet50_hmr_penn_action.py](resnet50_hmr_penn_action.py) | 114.53  |
| 3D-People | [resnet50_hmr_people3d.py](resnet50_hmr_people3d.py) | 118.31    |
| PoseTrack | [resnet50_hmr_posetrack.py](resnet50_hmr_posetrack.py) | 105.30 |
| PROX | [resnet50_hmr_prox.py](resnet50_hmr_prox.py) | 84.69 |
| 3DPW | [resnet50_hmr_pw3d.py](resnet50_hmr_pw3d.py) | 89.36 |
| SURREAL | [resnet50_hmr_surreal.py](resnet50_hmr_surreal.py) | 110.00  |
| UP-3D | [resnet50_hmr_up3d.py](resnet50_hmr_up3d.py) | 86.92 |
| VLOG | [resnet50_hmr_vlog.py](resnet50_hmr_vlog.py) | 100.38  |
