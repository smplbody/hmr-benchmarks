import glob
import os
import xml.etree.ElementTree as ET

import cdflib
import cv2
import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameter import CameraParameter
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


class H36mCamera():

    def __init__(self, metadata):
        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.metadata = metadata
        self.camera_ids = []
        self._load_metadata()
        self.image_sizes = {
            '54138969': {
                'width': 1000,
                'height': 1002
            },
            '55011271': {
                'width': 1000,
                'height': 1000
            },
            '58860488': {
                'width': 1000,
                'height': 1000
            },
            '60457274': {
                'width': 1000,
                'height': 1002
            }
        }

    def _load_metadata(self):
        """Load meta data from metadata.xml."""

        assert os.path.exists(self.metadata)

        tree = ET.parse(self.metadata)
        root = tree.getroot()

        for i, tr in enumerate(root.find('mapping')):
            if i == 0:
                _, _, *self.subjects = [td.text for td in tr]
                self.sequence_mappings \
                    = {subject: {} for subject in self.subjects}
            elif i < 33:
                action_id, subaction_id, *prefixes = [td.text for td in tr]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id, subaction_id)]\
                        = prefix

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        self.camera_ids \
            = [elem.text for elem in root.find('dbcameras/index2id')]

        w0 = root.find('w0')
        self.cameras_raw = [float(num) for num in w0.text[1:-1].split()]

    @staticmethod
    def get_intrinsic_matrix(f, c, inv=False):
        intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)
        intrinsic_metrix[0, 0] = f[0]
        intrinsic_metrix[0, 2] = c[0]
        intrinsic_metrix[1, 1] = f[1]
        intrinsic_metrix[1, 2] = c[1]
        intrinsic_metrix[2, 2] = 1

        if inv:
            intrinsic_metrix = np.linalg.inv(intrinsic_metrix).astype(
                np.float32)
        return intrinsic_metrix

    def _get_camera_params(self, camera, subject):
        """Get camera parameters given camera id and subject id."""
        metadata_slice = np.zeros(15)
        start = 6 * (camera * 11 + (subject - 1))

        metadata_slice[:6] = self.cameras_raw[start:start + 6]
        metadata_slice[6:] = self.cameras_raw[265 + camera * 9 - 1:265 +
                                              (camera + 1) * 9 - 1]

        # extrinsics
        x, y, z = -metadata_slice[0], metadata_slice[1], -metadata_slice[2]

        R_x = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)],
                        [0, -np.sin(x), np.cos(x)]])
        R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])
        R_z = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z),
                                                    np.cos(z), 0], [0, 0, 1]])
        R = (R_x @ R_y @ R_z).T
        T = metadata_slice[3:6].reshape(-1)
        # convert unit from millimeter to meter
        T *= 0.001

        # intrinsics
        c = metadata_slice[8:10, None]
        f = metadata_slice[6:8, None]
        K = self.get_intrinsic_matrix(f, c)

        # distortion
        k = metadata_slice[10:13, None]
        p = metadata_slice[13:15, None]

        w = self.image_sizes[self.camera_ids[camera]]['width']
        h = self.image_sizes[self.camera_ids[camera]]['height']

        camera_name = f'S{subject}_{self.camera_ids[camera]}'
        camera_params = CameraParameter(camera_name, h, w)
        camera_params.set_K_R_T(K, R, T)
        camera_params.set_value('k1', k[0])
        camera_params.set_value('k2', k[1])
        camera_params.set_value('k3', k[2])
        camera_params.set_value('p1', p[0])
        camera_params.set_value('p2', p[1])
        return camera_params.to_dict()

    def generate_cameras_dict(self):
        """Generate dictionary of camera params which contains camera
        parameters for 11 subjects each with 4 cameras."""
        cameras = {}
        for subject in range(1, 12):
            for camera in range(4):
                key = (f'S{subject}', self.camera_ids[camera])
                cameras[key] = self._get_camera_params(camera, subject)

        return cameras


@DATA_CONVERTERS.register_module()
class H36mConverter(BaseModeConverter):

    ACCEPTED_MODES = ['valid', 'train']

    def __init__(self, modes=[], protocol=1, extract_img=False):
        super(H36mConverter, self).__init__(modes)
        accepted_protocol = [1, 2]
        if protocol not in accepted_protocol:
            raise ValueError('Input protocol not in accepted protocol. \
                Use either 1 or 2')
        self.protocol = protocol
        self.extract_img = extract_img
        self.camera_name_to_idx = {
            '54138969': 0,
            '55011271': 1,
            '58860488': 2,
            '60457274': 3,
        }

    def convert_by_mode(self, dataset_path, out_path, mode):
        # use HumanData to store all data
        human_data = HumanData()

        # pick 17 joints from 32 (repeated) joints
        h36m_idx = [
            11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27
        ]

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []

        # choose users ids for different set
        if mode == 'train':
            user_list = [1, 5, 6, 7, 8]
        elif mode == 'valid':
            user_list = [9, 11]

        # go over each user
        for user_i in tqdm(user_list, desc='user id'):
            user_name = f'S{user_i}'
            # path with GT bounding boxes
            bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                                     'ground_truth_bs')
            # path with GT 2D pose
            pose2d_path = os.path.join(dataset_path, user_name,
                                       'MyPoseFeatures', 'D2_Positions')
            # path with GT 3D pose
            pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                     'D3_Positions_mono')
            # path with videos
            vid_path = os.path.join(dataset_path, user_name, 'Videos')

            # go over all the sequences of each user
            seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
            seq_list.sort()

            for seq_i in tqdm(seq_list, desc='sequence id'):

                # sequence info
                seq_name = seq_i.split('/')[-1]
                action, camera, _ = seq_name.split('.')
                action = action.replace(' ', '_')
                # irrelevant sequences
                if action == '_ALL':
                    continue

                # 2D pose file
                pose2d_file = os.path.join(pose2d_path, seq_name)
                poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

                # 3D pose file
                poses_3d = cdflib.CDF(seq_i)['Pose'][0]

                # bbox file
                bbox_file = os.path.join(bbox_path,
                                         seq_name.replace('cdf', 'mat'))
                bbox_h5py = h5py.File(bbox_file)

                # video file
                if self.extract_img:
                    vid_file = os.path.join(vid_path,
                                            seq_name.replace('cdf', 'mp4'))
                    vidcap = cv2.VideoCapture(vid_file)

                # go over each frame of the sequence
                for frame_i in tqdm(range(poses_3d.shape[0]), desc='frame id'):
                    # read video frame
                    if self.extract_img:
                        success, image = vidcap.read()
                        if not success:
                            break

                    # check if you can keep this frame
                    if frame_i % 5 == 0 and (self.protocol == 1
                                             or camera == '60457274'):
                        # image name
                        seq_id = f'{user_name}_{action}'
                        image_name = f'{seq_id}.{camera}_{frame_i + 1:06d}.jpg'
                        img_folder_name = f'{user_name}_{action}.{camera}'
                        image_path = os.path.join(dataset_path, user_name,
                                                  'images', img_folder_name,
                                                  image_name)

                        # save image
                        if self.extract_img:
                            cv2.imwrite(image_path)

                        # read GT bounding box
                        mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]][:].T
                        ys, xs = np.where(mask == 1)
                        bbox_xyxy = np.array([
                            np.min(xs),
                            np.min(ys),
                            np.max(xs) + 1,
                            np.max(ys) + 1
                        ])
                        bbox_xywh = self._bbox_expand(
                            bbox_xyxy, scale_factor=0.9)

                        # read GT 2D pose
                        keypoints2dall = np.reshape(poses_2d[frame_i, :],
                                                    [-1, 2])
                        keypoints2d17 = keypoints2dall[h36m_idx]
                        keypoints2d17 = np.concatenate(
                            [keypoints2d17, np.ones((17, 1))], axis=1)

                        # read GT 3D pose
                        keypoints3dall = np.reshape(poses_3d[frame_i, :],
                                                    [-1, 3]) / 1000.
                        keypoints3d17 = keypoints3dall[h36m_idx]
                        keypoints3d17 -= keypoints3d17[0]  # root-centered
                        keypoints3d17 = np.concatenate(
                            [keypoints3d17, np.ones((17, 1))], axis=1)

                        # store data
                        image_path_.append(image_path)
                        bbox_xywh_.append(bbox_xywh)
                        keypoints2d_.append(keypoints2d17)
                        keypoints3d_.append(keypoints3d17)

        metadata_path = os.path.join(dataset_path, 'metadata.xml')
        if isinstance(metadata_path, str):
            cam_param = H36mCamera(metadata_path)
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'h36m', 'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 17, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'h36m', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['cam_param'] = cam_param
        human_data['config'] = 'h36m'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if mode == 'train':
            out_file = os.path.join(out_path, 'h36m_train.npz')
        elif mode == 'valid':
            out_file = os.path.join(out_path,
                                    f'h36m_valid_protocol{self.protocol}.npz')
        human_data.dump(out_file)
