import json
import os
from typing import List

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class Muco3dhpConverter(BaseConverter):
    """SURREAL dataset `Learning from Synthetic Humans' CVPR`2017 More details
    can be found in the `paper.

    <https://arxiv.org/pdf/1701.01370.pdf>`__.

    Args:
        modes (list): 'val', 'test' or 'train' for accepted modes
        run (int): 0, 1, 2 for available runs
        extract_img (bool): Store True to extract images from video.
        Default: False.
    """
    # ACCEPTED_MODES = ['val', 'train', 'test']

    # def __init__(self,
    #              modes: List = [],
    #              run: int = 0,
    #              extract_img: bool = False) -> None:
    #     super(Muco3dhpConverter, self).__init__(modes)
    #     accepted_runs = [0, 1, 2]
    #     if run not in accepted_runs:
    #         raise ValueError('Input run not in accepted runs. \
    #             Use either 0 or 1 or 2')
    #     self.run = run
    #     self.extract_img = extract_img
    #     self.image_height = 1080
    #     self.image_width = 1920
    @staticmethod
    def get_intrinsic_matrix(f: List[float],
                             c: List[float],
                             inv: bool = False) -> np.ndarray:
        """Get intrisic matrix (or its inverse) given f and c."""
        intrinsic_matrix = np.zeros((3, 3)).astype(np.float32)
        intrinsic_matrix[0, 0] = f[0]
        intrinsic_matrix[0, 2] = c[0]
        intrinsic_matrix[1, 1] = f[1]
        intrinsic_matrix[1, 2] = c[1]
        intrinsic_matrix[2, 2] = 1

        if inv:
            intrinsic_matrix = np.linalg.inv(intrinsic_matrix).astype(
                np.float32)
        return intrinsic_matrix

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask, video_path,
                smpl, meta, cam_param stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_, \
            cam_param_ = [], [], [], [], []

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []
        # smpl['transl'] = []
        # meta = {}
        # meta['gender'] = []

        # data_path = os.path.join(dataset_path,
        #                          '{}/run{}'.format(mode, self.run)
        annot_path = dataset_path.replace('muco', 'extras/MuCo-3DHP.json')
        smpl_param_path = os.path.join(dataset_path, 'SMPLX/smpl_param.json')

        db = COCO(annot_path)
        with open(smpl_param_path) as f:
            smpl_params = json.load(f)

        # datalist = []
        for iid in tqdm(db.imgs.keys()):
            img = db.imgs[iid]
            # img_id = img["id"]
            w, h = img['width'], img['height']
            imgname = img['file_name']
            if 'unaugmented_set' not in imgname:

                # img_path = os.path.join(self.img_dir, imgname)
                # focal = img['f']
                # princpt = img['c']
                R = np.array(img['R']).reshape(3, 3)
                T = np.array(img['T']).reshape(3, )
                K = self.get_intrinsic_matrix(img['f'], img['c'])
                # K = get_intrinsic_from_fc(img['f'], img['c'])
                # cam_param = {'focal': focal, 'princpt': princpt}

                # crop the closest person to the camera
                ann_ids = db.getAnnIds(img['id'])
                anns = db.loadAnns(ann_ids)

                camera = CameraParameter(H=h, W=w)
                camera.set_KRT(K, R, T)
                parameter_dict = camera.to_dict()

                for i, pid in enumerate(ann_ids):
                    try:
                        smpl_param = smpl_params[str(pid)]
                        pose, shape, trans = np.array(
                            smpl_param['pose']), np.array(
                                smpl_param['shape']), np.array(
                                    smpl_param['trans'])
                        sum = pose.sum() + shape.sum() + trans.sum()
                        if np.isnan(sum):
                            continue
                    except KeyError:
                        continue

                    joint_cam = np.array(anns[i]['keypoints_cam'])
                    joint_img = np.array(anns[i]['keypoints_img'])

                    # keypoints3d = joint_cam / 1000
                    joint_cam = joint_cam - joint_cam[14]  # 4 is the root

                    bbox = np.array(anns[i]['bbox'])
                    keypoints_vis = np.array(
                        anns[i]['keypoints_vis']).astype('int').reshape(-1, 1)
                    if not int(keypoints_vis[14]) == 1:
                        continue
                    joint_img = np.hstack([joint_img, keypoints_vis])
                    joint_cam = np.hstack([joint_cam, keypoints_vis])

                    keypoints2d_.append(joint_img)
                    keypoints3d_.append(joint_cam)
                    bbox_xywh_.append(bbox)
                    smpl['body_pose'].append(pose[3:].reshape((23, 3)))
                    smpl['global_orient'].append(pose[:3])
                    smpl['betas'].append(shape)
                    # smpl['transl'].append(trans)
                    # meta['gender'].append(gender)
                    cam_param_.append(parameter_dict)
                    image_path_.append(f'images/{imgname}')

        # change list to np array
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        # meta['gender'] = np.array(meta['gender']).reshape(-1)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 21, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'muco', 'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 21, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'muco', 'human_data')

        human_data['image_path'] = image_path_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints3d'] = keypoints3d_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'muco3dhp'
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'muco3dhp_augmented_train.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
