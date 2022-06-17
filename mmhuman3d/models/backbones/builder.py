# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import PoseHighResolutionNet
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .swin import SwinTransformer
from .twins import PCPVT, SVT
from .vit import VisionTransformer

BACKBONES = Registry('backbones')

BACKBONES.register_module(name='ResNet', module=ResNet)
BACKBONES.register_module(name='ResNetV1d', module=ResNetV1d)
BACKBONES.register_module(
    name='PoseHighResolutionNet', module=PoseHighResolutionNet)
BACKBONES.register_module(name='EfficientNet', module=EfficientNet)
BACKBONES.register_module(name='HourglassNet', module=HourglassNet)
BACKBONES.register_module(name='ResNeXt', module=ResNeXt)
BACKBONES.register_module(name='SwinTransformer', module=SwinTransformer)
BACKBONES.register_module(name='SVT', module=SVT)
BACKBONES.register_module(name='PCPVT', module=PCPVT)
BACKBONES.register_module(name='VisionTransformer', module=VisionTransformer)


def build_backbone(cfg):
    """Build backbone."""
    if cfg is None:
        return None
    return BACKBONES.build(cfg)
