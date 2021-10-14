import glob
import os

import cv2
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class MpiInf3dhpConverter(BaseModeConverter):
    ACCEPTED_MODES = ['test', 'train']

    def __init__(self, modes=[], extract_img=False):
        super(MpiInf3dhpConverter, self).__init__(modes)
        self.extract_img = extract_img

    def extract_keypoints(self, keypoints2d, keypoints3d, num_keypoints):

        bbox_xyxy = [
            min(keypoints2d[:, 0]),
            min(keypoints2d[:, 1]),
            max(keypoints2d[:, 0]),
            max(keypoints2d[:, 1])
        ]
        bbox_xywh = self._bbox_expand(bbox_xyxy, scale_factor=1.2)

        # check that all joints are visible
        h, w = 2048, 2048
        x_in = np.logical_and(keypoints2d[:, 0] < w, keypoints2d[:, 0] >= 0)
        y_in = np.logical_and(keypoints2d[:, 1] < h, keypoints2d[:, 1] >= 0)
        ok_pts = np.logical_and(x_in, y_in)
        if np.sum(ok_pts) < num_keypoints:
            valid = False

        # add confidence column
        keypoints2d = np.hstack([keypoints2d, np.ones((num_keypoints, 1))])
        keypoints3d = np.hstack([keypoints3d, np.ones((num_keypoints, 1))])
        valid = True

        return valid, keypoints2d, keypoints3d, bbox_xywh

    def convert_by_mode(self, dataset_path, out_path, mode):

        total_dict = {}

        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []

        # training data
        if mode == 'train':
            user_list = range(1, 9)
            seq_list = range(1, 3)
            vid_list = list(range(3)) + list(range(4, 9))

            counter = 0

            for user_i in tqdm(user_list, desc='user list'):
                for seq_i in seq_list:
                    seq_path = os.path.join(dataset_path, 'S' + str(user_i),
                                            'Seq' + str(seq_i))
                    # mat file with annotations
                    annot_file = os.path.join(seq_path, 'annot.mat')
                    annot2 = sio.loadmat(annot_file)['annot2']
                    annot3 = sio.loadmat(annot_file)['annot3']

                    for j, vid_i in tqdm(enumerate(vid_list), desc='vid list'):

                        # image folder
                        imgs_path = os.path.join(seq_path,
                                                 'video_' + str(vid_i))

                        # extract frames from video file
                        if self.extract_img:

                            # if doesn't exist
                            if not os.path.isdir(imgs_path):
                                os.makedirs(imgs_path)

                            # video file
                            vid_file = os.path.join(
                                seq_path, 'imageSequence',
                                'video_' + str(vid_i) + '.avi')
                            vidcap = cv2.VideoCapture(vid_file)

                            # process video
                            frame = 0
                            while 1:
                                # extract all frames
                                success, image = vidcap.read()
                                if not success:
                                    break
                                frame += 1
                                # image name
                                imgname = os.path.join(
                                    imgs_path, 'frame_%06d.jpg' % frame)
                                # save image
                                cv2.imwrite(imgname, image)

                        # per frame
                        pattern = os.path.join(imgs_path, '*.jpg')
                        img_list = glob.glob(pattern)
                        for i, img_i in enumerate(sorted(img_list)):

                            # for each image we store the relevant annotations
                            img_name = img_i.split('/')[-1]
                            image_path = os.path.join('S' + str(user_i),
                                                      'Seq' + str(seq_i),
                                                      'video_' + str(vid_i),
                                                      img_name)

                            # 2D keypoints
                            keypoints2d = np.reshape(annot2[vid_i][0][i],
                                                     (28, 2))

                            # 3D keypoints
                            keypoints3d = np.reshape(annot3[vid_i][0][i],
                                                     (28, 3)) / 1000
                            keypoints3d = keypoints3d - keypoints3d[
                                4]  # 4 is the root
                            valid, keypoints2d, keypoints3d, bbox_xywh = \
                                self.extract_keypoints(
                                    keypoints2d, keypoints3d, 28)

                            if not valid:
                                continue

                            # because of the dataset size,
                            # we only keep every 10th frame
                            counter += 1
                            if counter % 10 != 1:
                                continue

                            # store the data
                            image_path_.append(image_path)
                            bbox_xywh_.append(bbox_xywh)
                            keypoints2d_.append(keypoints2d)
                            keypoints3d_.append(keypoints3d)

            keypoints2d_ = np.array(keypoints2d_).reshape((-1, 28, 3))
            keypoints2d_, mask = convert_kps(keypoints2d_, 'mpi_inf_3dhp',
                                             'smplx')
            keypoints3d_ = np.array(keypoints3d_).reshape((-1, 28, 4))
            keypoints3d_, _ = convert_kps(keypoints3d_, 'mpi_inf_3dhp',
                                          'smplx')

        elif mode == 'test':

            # test data
            user_list = range(1, 7)

            for user_i in tqdm(user_list, desc='user'):
                seq_path = os.path.join(dataset_path, 'mpi_inf_3dhp_test_set',
                                        'TS' + str(user_i))
                # mat file with annotations
                annot_file = os.path.join(seq_path, 'annot_data.mat')
                mat_as_h5 = h5py.File(annot_file, 'r')
                annot2 = np.array(mat_as_h5['annot2'])
                annot3 = np.array(mat_as_h5['univ_annot3'])
                valid = np.array(mat_as_h5['valid_frame'])
                for frame_i, valid_i in tqdm(enumerate(valid), desc='frame'):

                    if valid_i == 0:
                        continue
                    image_path = os.path.join(
                        'mpi_inf_3dhp_test_set', 'TS' + str(user_i),
                        'imageSequence',
                        'img_' + str(frame_i + 1).zfill(6) + '.jpg')
                    keypoints2d = annot2[frame_i, 0, :, :]
                    keypoints3d = annot3[frame_i, 0, :, :] / 1000
                    keypoints3d = keypoints3d - keypoints3d[14]  # 14 is pelvis

                    valid, keypoints2d, keypoints3d, bbox_xywh = \
                        self.extract_keypoints(keypoints2d, keypoints3d, 17)

                    if not valid:
                        continue

                    # store the data
                    image_path_.append(image_path)
                    bbox_xywh_.append(bbox_xywh)
                    keypoints2d_.append(keypoints2d)
                    keypoints3d_.append(keypoints3d)

            keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
            keypoints2d_, mask = convert_kps(keypoints2d_, 'mpi_inf_3dhp_test',
                                             'smplx')
            keypoints3d_ = np.array(keypoints3d_).reshape((-1, 17, 4))
            keypoints3d_, _ = convert_kps(keypoints3d_, 'mpi_inf_3dhp_test',
                                          'smplx')

        total_dict['image_path'] = image_path_
        total_dict['bbox_xywh'] = bbox_xywh_
        total_dict['keypoints2d'] = keypoints2d_
        total_dict['keypoints3d'] = keypoints3d_
        total_dict['mask'] = mask
        total_dict['config'] = 'mpi_inf_3dhp'

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'mpi_inf_3dhp_{}.npz'.format(mode))
        np.savez_compressed(out_file, **total_dict)
