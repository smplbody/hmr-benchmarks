import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class LIPConverter(BaseModeConverter):
    """CocoWholeBodyDataset dataset `Whole-Body Human Pose Estimation in the
    Wild' ECCV'2020 More details can be found in the `paper.

    <https://arxiv.org/abs/2007.11858>`__ .

    Args:
        modes (list): 'val' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['val', 'train']

    def __init__(self, modes: List = []) -> None:
        super(LIPConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        df = pd.read_csv(
            os.path.join(dataset_path, f'lip_{mode}_set.csv'), header=None)
        df = df.fillna(1)

        for img_name, kp in tqdm(
                zip(df.iloc[:, 0].values, df.iloc[:, 1:].values),
                total=len(df)):
            img_path = os.path.join(f'{mode}_images', str(img_name))
            keypoints2d = np.reshape(kp, (16, 3))
            keypoints2d[:, 2] = 1 - keypoints2d[:, 2]

            vis_index = np.where(keypoints2d[:, 2] > 0)[0]
            if len(vis_index) < 5:
                continue
            vis_keypoints2d = keypoints2d[vis_index]
            # bbox
            bbox_xyxy = [
                min(vis_keypoints2d[:, 0]),
                min(vis_keypoints2d[:, 1]),
                max(vis_keypoints2d[:, 0]),
                max(vis_keypoints2d[:, 1])
            ]
            bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
            bbox_xywh = self._xyxy2xywh(bbox_xyxy)

            # store data
            image_path_.append(img_path)
            keypoints2d_.append(keypoints2d)
            bbox_xywh_.append(bbox_xywh)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 16, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'lip', 'human_data')

        human_data['image_path'] = image_path_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['config'] = 'lip'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'lip_{}.npz'.format(mode))
        human_data.dump(out_file)
