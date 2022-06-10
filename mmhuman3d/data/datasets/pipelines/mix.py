import math
import random

import albumentations
import mmcv
import numpy as np

from ..builder import PIPELINES


def get_random_crop_coords(height, width, crop_height, crop_width, h_start,
                           w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(center, scale, crop_scale_factor, axis='all'):
    '''
    center: bbox center [x,y]
    scale: bbox height / 200
    crop_scale_factor: amount of cropping to be applied
    axis: axis which cropping will be applied
        "x": center the y axis and get random crops in x
        "y": center the x axis and get random crops in y
        "all": randomly crop from all locations
    '''
    orig_size = int(scale[0])
    ul = (center - (orig_size / 2.)).astype(int)

    crop_size = int(orig_size * crop_scale_factor)

    if axis == 'all':
        h_start = np.random.rand()
        w_start = np.random.rand()
    elif axis == 'x':
        h_start = np.random.rand()
        w_start = 0.5
    elif axis == 'y':
        h_start = 0.5
        w_start = np.random.rand()
    else:
        raise ValueError(f'axis {axis} is undefined!')

    x1, y1, x2, y2 = get_random_crop_coords(
        height=orig_size,
        width=orig_size,
        crop_height=crop_size,
        crop_width=crop_size,
        h_start=h_start,
        w_start=w_start,
    )
    scale = y2 - y1
    center = ul + np.array([(y1 + y2) / 2, (x1 + x2) / 2])
    return center, np.array([scale, scale])


@PIPELINES.register_module()
class RandomCrop:
    """Data augmentation with random channel noise.

    # numpy array with uint8
    Required keys: 'img'
    Modifies key: 'img'
    Args:
        noise_factor (float): Multiply each channel with
         a factor between``[1-scale_factor, 1+scale_factor]``
    """

    def __init__(self, crop_prob=0.4, crop_factor=0.2, axis='y'):
        self.crop_prob = crop_prob
        self.crop_factor = crop_factor
        self.axis = axis

    def __call__(self, results):
        """Perform data augmentation with random channel noise."""

        if np.random.rand() <= self.crop_prob:
            c = results['center']
            s = results['scale']
            center, scale = random_crop(
                c, s, crop_scale_factor=1 - self.crop_factor, axis=self.axis)
            results['center'] = center
            results['scale'] = scale
        return results


@PIPELINES.register_module()
class Albumentation:
    """Albumentation augmentation (pixel-level transforms only). Adds custom
    pixel-level transformations from Albumentations library. Please visit
    `https://albumentations.readthedocs.io` to get more information.
    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.
    An example of ``transforms`` is as followed:
    .. code-block:: python
        [
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of Albumentation transformations
        keymap (dict): Contains {'input key':'albumentation-style key'},
            e.g., {'img': 'image'}.
    """

    def __init__(self, transforms, keymap=None):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {keymap.get(k, k): v for k, v in d.items()}
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)
        # back to the original format
        results = self.mapper(results, self.keymap_back)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class PhotometricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5.

    The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 mlist=None):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.mlist = mlist

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beta with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(-self.brightness_delta,
                                       self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                alpha=np.random.uniform(self.contrast_lower,
                                        self.contrast_upper))
        return img

    def saturation(self, img):
        # Apply saturation distortion to hsv-formatted img
        img[:, :, 1] = self.convert(
            img[:, :, 1],
            alpha=np.random.uniform(self.saturation_lower,
                                    self.saturation_upper))
        return img

    def hue(self, img):
        # Apply hue distortion to hsv-formatted img
        img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  np.random.randint(-self.hue_delta, self.hue_delta)) % 180
        return img

    def swap_channels(self, img):
        # Apply channel swap
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        if results['dataset_name'] not in self.mlist:
            return results

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        hsv_mode = np.random.randint(4)
        if hsv_mode:
            # random saturation/hue distortion
            img = mmcv.bgr2hsv(img)
            if hsv_mode == 1 or hsv_mode == 3:
                img = self.saturation(img)
            if hsv_mode == 2 or hsv_mode == 3:
                img = self.hue(img)
            img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        # randomly swap channels
        self.swap_channels(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class TopDownRandomShiftBboxCenter:
    """Random shift the bbox center.

    Required key: 'center', 'scale'
    Modifies key: 'center'
    Args:
        shift_factor (float): The factor to control the shift range, which is
            scale*pixel_std*scale_factor. Default: 0.16
        prob (float): Probability of applying random shift. Default: 0.3
    """

    # Pixel std is 200.0, which serves as the normalization factor to
    # to calculate bbox scales.
    # pixel_std: float = 200.0

    def __init__(self, shift_factor: float = 0.16, prob: float = 0.3):
        self.shift_factor = shift_factor
        self.prob = prob

    def __call__(self, results):

        center = results['center']
        scale = results['scale']
        if np.random.rand() < self.prob:
            center += np.random.uniform(-1, 1, 2) * self.shift_factor * scale

        results['center'] = center
        return results


@PIPELINES.register_module()
class MixingErasing:
    """Randomly selects a rectangle region in an image and erases its pixels
    with different mixing operation.

    normal: original random erasing;
    soft: mixing ori with random pixel;
    self: mixing ori with other_ori_patch;
    Args:
         probability: The probability that the Random Erasing operation
            will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(
            self,
            probability=0.5,
            sl=0.02,
            sh=0.4,
            r1=0.3,
            #  mean=(0.4914, 0.4822, 0.4465),
            mode='pixel',  # pixel
            ntype='normal',  # normal
            mixing_coeff=[1.0, 1.0],
            mlist=None):
        self.probability = probability
        # self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        # self.device = device
        self.ntype = ntype
        self.mixing_coeff = mixing_coeff
        self.mlist = mlist

    def __call__(self, results):
        """Perform data augmentation with random channel noise."""

        if results['dataset_name'] not in self.mlist:
            return results

        if np.random.rand() > self.probability:
            return results

        img = results['img']
        img_h, img_w, _ = img.shape

        for attempt in range(100):
            area = img_h * img_w

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                x1 = random.randint(0, img_h - h)  # xmin
                y1 = random.randint(0, img_w - w)  # ymin
                m = random.uniform(0.1, 0.9)
                if self.ntype == 'self':
                    x2 = random.randint(0, img_h - h)
                    y2 = random.randint(0, img_w - w)
                    img[x1:x1 + h, y1:y1 +
                        w, :] = (1 - m) * img[x1:x1 + h, y1:y1 +
                                              w, :] + m * img[x2:x2 + h,
                                                              y2:y2 + w, :]
                else:
                    if self.mode == 'const':
                        img[x1:x1 + h,
                            y1:y1 + w, :] = np.random.rand(h, w, 3) * 255
                    else:
                        img[x1:x1 + h, y1:y1 +
                            w, :] = (1 - m) * img[x1:x1 + h, y1:y1 +
                                                  w, :] + m * np.random.rand(
                                                      h, w, 3) * 255

                results['img'] = img
                break

        return results


def center_scale_to_xyxy(center, scale):
    """obtain bbox from center and scale with pixel_std=200."""
    # w, h = scale * 200, scale * 200
    w, h = scale[0], scale[1]
    x1, y1 = center[0] - w / 2, center[1] - h / 2
    x2, y2 = center[0] + w / 2, center[1] + h / 2
    return [x1, y1, x2, y2]


@PIPELINES.register_module()
class HardErasing:
    """Add random occlusion.

    Add random occlusion based on occlusion probability.

    Args:
        occlusion_prob (float): probability of the image having
        occlusion. Default: 0.5
    """

    def __init__(self, occlusion_prob=0.5):
        self.occlusion_prob = occlusion_prob

    def __call__(self, results):

        if np.random.rand() > self.occlusion_prob:
            return results
        c = results['center']
        s = results['scale']
        xmin, ymin, xmax, ymax = center_scale_to_xyxy(c, s)
        imgheight, imgwidth, _ = results['img'].shape

        img = results['img']

        area_min = 0.0
        area_max = 0.7
        synth_area = (random.random() *
                      (area_max - area_min) + area_min) * (xmax - xmin) * (
                          ymax - ymin)

        ratio_min = 0.3
        ratio_max = 1 / 0.3
        synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

        synth_h = math.sqrt(synth_area * synth_ratio)
        synth_w = math.sqrt(synth_area / synth_ratio)
        synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
        synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

        if synth_xmin >= 0 and synth_ymin >= 0 and \
            synth_xmin + synth_w < imgwidth and \
                synth_ymin + synth_h < imgheight:
            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)
            img[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin +
                synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255

        results['img'] = img

        return results
