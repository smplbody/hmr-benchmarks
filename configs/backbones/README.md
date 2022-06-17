# Backbones

## Introduction

We provide the config files for training on different backbones:


```BibTeX
@inproceedings{HMR,
  author    = {Angjoo Kanazawa and
               Michael J. Black and
               David W. Jacobs and
               Jitendra Malik},
  title     = {End-to-End Recovery of Human Shape and Pose},
  booktitle = {CVPR},
  year      = {2018}
}
```


1. ResNet-50, -101, -152
<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

</details>

2. EfficientNet

<details>
<summary align="right"><a href="https://arxiv.org/abs/1905.11946v5">EfficientNet (ICML'2019)</a></summary>

```bibtex
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```
</details>

3. HRNet-W32

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

4. ResNext

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html">ResNext (CVPR'2017)</a></summary>

```bibtex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}
```

</details>


5. ViT

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2010.11929.pdf">ViT (ICLR'2021)</a></summary>

```bibtex
@inproceedings{dosovitskiy2021an,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=YicbFdNTTy}
}
```


6. Swin

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2103.14030.pdf">Swin (ICCV'2021)</a></summary>

```bibtex
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

7. Twins-PCVCT, -SVT

<details>
<summary align="right"><a href="http://arxiv-export-lb.library.cornell.edu/abs/2104.13840">Twins (NeurIPS'2021)</a></summary>

```bibtex
@inproceedings{chu2021Twins,
	title={Twins: Revisiting the Design of Spatial Attention in Vision Transformers},
	author={Xiangxiang Chu and Zhi Tian and Yuqing Wang and Bo Zhang and Haibing Ren and Xiaolin Wei and Huaxia Xia and Chunhua Shen},
	booktitle={NeurIPS 2021},
	year={2021}
}
```


## Results and Models

We evaluate HMR on 3DPW. Values are MPJPE/PA-MPJPE.

| Config | 3DPW    |
|:------:|:-------:|
| [resnet50_hmr_pw3d.py](resnet50_hmr_pw3d.py) | 112.46 / 64.55 |
| [resnet101_hmr_pw3d.py](resnet101_hmr_pw3d.py) | 112.67 / 63.36 |
| [resnet152_hmr_pw3d.py](resnet152_hmr_pw3d.py) | 107.13 / 62.13 |
| [resnext101_hmr_pw3d.py](resnext101_hmr_pw3d.py) | 114.43 / 64.95 |
| [efficientnet_hmr_pw3d.py](efficientnet_hmr_pw3d.py) | 112.34 / 67.53 |
| [hrnet_hmr_pw3d.py](hrnet_hmr_pw3d.py) | 118.15 / 65.16 |
| [vit_hmr_pw3d.py](vit_hmr_pw3d.py) | 111.46 / 62.81 |
| [swin_hmr_pw3d.py](swin_hmr_pw3d.py) | 110.42 / 62.78 |
| [twins_pcpvt_hmr_pw3d.py](twins_pcpvt_hmr_pw3d.py) | 100.75 / 59.13 |
| [twins_svt_hmr_pw3d.py](twins_svt_hmr_pw3d.py) | 105.42 / 60.11 |
