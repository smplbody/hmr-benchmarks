from collections import defaultdict

HUMAN_DATA = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine_1',
    'left_knee',
    'right_knee',
    'spine_2',
    'left_ankle',
    'right_ankle',
    'spine_3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eyeball',
    'right_eyeball',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_bigtoe',
    'left_smalltoe',
    'left_heel',
    'right_bigtoe',
    'right_smalltoe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eyebrow_1',
    'right_eyebrow_2',
    'right_eyebrow_3',
    'right_eyebrow_4',
    'right_eyebrow_5',
    'left_eyebrow_5',
    'left_eyebrow_4',
    'left_eyebrow_3',
    'left_eyebrow_2',
    'left_eyebrow_1',
    'nosebridge_1',
    'nosebridge_2',
    'nosebridge_3',
    'nosebridge_4',
    'nose_1',
    'nose_2',
    'nose_3',
    'nose_4',
    'nose_5',
    'right_eye_1',
    'right_eye_2',
    'right_eye_3',
    'right_eye_4',
    'right_eye_5',
    'right_eye_6',
    'left_eye_4',
    'left_eye_3',
    'left_eye_2',
    'left_eye_1',
    'left_eye_6',
    'left_eye_5',
    'mouth_1',
    'mouth_2',
    'mouth_3',
    'mouth_4',
    'mouth_5',
    'mouth_6',
    'mouth_7',
    'mouth_8',
    'mouth_9',
    'mouth_10',
    'mouth_11',
    'mouth_12',
    'lip_1',
    'lip_2',
    'lip_3',
    'lip_4',
    'lip_5',
    'lip_6',
    'lip_7',
    'lip_8',
    'face_contour_1',
    'face_contour_2',
    'face_contour_3',
    'face_contour_4',
    'face_contour_5',
    'face_contour_6',
    'face_contour_7',
    'face_contour_8',
    'face_contour_9',
    'face_contour_10',
    'face_contour_11',
    'face_contour_12',
    'face_contour_13',
    'face_contour_14',
    'face_contour_15',
    'face_contour_16',
    'face_contour_17',
    # J_regressor_extra
    'right_hip_extra',
    'left_hip_extra',
    'neck_extra',  # LSP
    'headtop',  # LSP mpii peen_action mpi_inf_3dhp
    'pelvis_extra',  # MPII
    'thorax_extra',  # MPII
    'spine_extra',  # H36M
    'jaw_extra',  # H36M
    'head_extra',  # H36M
    # openpose
    'nose_openpose',
    'neck_openpose',
    'right_shoulder_openpose',
    'right_elbow_openpose',
    'right_wrist_openpose',
    'left_shoulder_openpose',
    'left_elbow_openpose',
    'left_wrist_openpose',
    'pelvis_openpose',
    'right_hip_openpose',
    'right_knee_openpose',
    'right_ankle_openpose',
    'left_hip_openpose',
    'left_knee_openpose',
    'left_ankle_openpose',
    'right_eye_openpose',
    'left_eye_openpose',
    'right_ear_openpose',
    'left_ear_openpose',
    'left_bigtoe_openpose',
    'left_smalltoe_openpose',
    'left_heel_openpose',
    'right_bigtoe_openpose',
    'right_smalltoe_openpose',
    'right_heel_openpose',
    # 3dhp
    'spine_4_3dhp',
    'left_clavicle_3dhp',
    'right_clavicle_3dhp',
    'left_hand_3dhp',
    'right_hand_3dhp',
    'left_toe_3dhp',
    'right_toe_3dhp',
    'head_h36m',  # H36M GT
    'headtop_h36m',  # H36M GT
    'head_bottom_pt',  # pose track
    'left_hand',  # SMPL
    'right_hand',  # SMPL
]

APPROXIMATE_MAPPING_LIST = [
    # extra
    ['pelvis', 'pelvis_openpose', 'pelvis_extra'],
    ['left_hip', 'left_hip_openpose', 'left_hip_extra'],
    ['right_hip', 'right_hip_openpose', 'right_hip_extra'],
    ['neck', 'neck_openpose', 'neck_extra'],
    ['jaw', 'jaw_extra'],
    ['head_extra', 'head_h36m'],
    ['headtop', 'headtop_h36m'],
    # 3dhp
    ['left_hand', 'left_hand_3dhp'],
    ['right_hand', 'right_hand_3dhp'],
    # openpose
    ['nose', 'nose_openpose'],
    ['right_shoulder', 'right_shoulder_openpose'],
    ['right_elbow', 'right_elbow_openpose'],
    ['right_wrist', 'right_wrist_openpose'],
    ['left_shoulder', 'left_shoulder_openpose'],
    ['left_elbow', 'left_elbow_openpose'],
    ['left_wrist', 'left_wrist_openpose'],
    ['right_knee', 'right_knee_openpose'],
    ['right_ankle', 'right_ankle_openpose'],
    ['left_knee', 'left_knee_openpose'],
    ['left_ankle', 'left_ankle_openpose'],
    ['right_eye', 'right_eye_openpose'],
    ['left_eye', 'left_eye_openpose'],
    ['right_ear', 'right_ear_openpose'],
    ['left_ear', 'left_ear_openpose'],
    ['left_bigtoe', 'left_bigtoe_openpose'],
    ['left_smalltoe', 'left_smalltoe_openpose'],
    ['left_heel', 'left_heel_openpose'],
    ['right_bigtoe', 'right_bigtoe_openpose'],
    ['right_smalltoe', 'right_smalltoe_openpose'],
    ['right_heel', 'right_heel_openpose'],
]

APPROXIMATE_MAP = defaultdict(list)
for group in APPROXIMATE_MAPPING_LIST:
    for member in group:
        for other_member in group:
            if member == other_member:
                continue
            APPROXIMATE_MAP[member].append(other_member)

HUMAN_DATA_HEAD = [
    'head', 'jaw', 'left_eyeball', 'right_eyeball', 'nose', 'right_eye',
    'left_eye', 'right_ear', 'left_ear', 'right_eyebrow_1', 'right_eyebrow_2',
    'right_eyebrow_3', 'right_eyebrow_4', 'right_eyebrow_5', 'left_eyebrow_5',
    'left_eyebrow_4', 'left_eyebrow_3', 'left_eyebrow_2', 'left_eyebrow_1',
    'nosebridge_1', 'nosebridge_2', 'nosebridge_3', 'nosebridge_4', 'nose_1',
    'nose_2', 'nose_3', 'nose_4', 'nose_5', 'right_eye_1', 'right_eye_2',
    'right_eye_3', 'right_eye_4', 'right_eye_5', 'right_eye_6', 'left_eye_4',
    'left_eye_3', 'left_eye_2', 'left_eye_1', 'left_eye_6', 'left_eye_5',
    'mouth_1', 'mouth_2', 'mouth_3', 'mouth_4', 'mouth_5', 'mouth_6',
    'mouth_7', 'mouth_8', 'mouth_9', 'mouth_10', 'mouth_11', 'mouth_12',
    'lip_1', 'lip_2', 'lip_3', 'lip_4', 'lip_5', 'lip_6', 'lip_7', 'lip_8',
    'face_contour_1', 'face_contour_2', 'face_contour_3', 'face_contour_4',
    'face_contour_5', 'face_contour_6', 'face_contour_7', 'face_contour_8',
    'face_contour_9', 'face_contour_10', 'face_contour_11', 'face_contour_12',
    'face_contour_13', 'face_contour_14', 'face_contour_15', 'face_contour_16',
    'face_contour_17', 'headtop', 'jaw_extra', 'head_extra', 'nose_openpose',
    'right_eye_openpose', 'left_eye_openpose', 'right_ear_openpose',
    'left_ear_openpose', 'headtop_h36m', 'head_bottom_pt', 'head_h36m'
]

HUMAN_DATA_LEFT_HAND = [
    'left_index_1', 'left_index_2', 'left_index_3', 'left_middle_1',
    'left_middle_2', 'left_middle_3', 'left_pinky_1', 'left_pinky_2',
    'left_pinky_3', 'left_ring_1', 'left_ring_2', 'left_ring_3',
    'left_thumb_1', 'left_thumb_2', 'left_thumb_3', 'left_thumb', 'left_index',
    'left_middle', 'left_ring', 'left_pinky', 'left_hand_3dhp', 'left_hand'
]

HUMAN_DATA_RIGHT_HAND = [
    'right_index_1', 'right_index_2', 'right_index_3', 'right_middle_1',
    'right_middle_2', 'right_middle_3', 'right_pinky_1', 'right_pinky_2',
    'right_pinky_3', 'right_ring_1', 'right_ring_2', 'right_ring_3',
    'right_thumb_1', 'right_thumb_2', 'right_thumb_3', 'right_thumb',
    'right_index', 'right_middle', 'right_ring', 'right_pinky',
    'right_hand_3dhp', 'right_hand'
]

HUMAN_DATA_SHOULDER = [
    'left_shoulder', 'left_shoulder_openpose', 'right_shoulder',
    'right_shoulder_openpose'
]

HUMAN_DATA_HIP = [
    'left_hip', 'left_hip_openpose', 'left_hip_extra', 'right_hip',
    'right_hip_openpose', 'right_hip_extra'
]

HUMAN_DATA_BODY = HUMAN_DATA_SHOULDER + HUMAN_DATA_HIP + [
    'pelvis', 'spine_1', 'left_knee', 'right_knee', 'spine_2', 'left_ankle',
    'right_ankle', 'spine_3', 'left_foot', 'right_foot', 'neck', 'left_collar',
    'right_collar', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_bigtoe', 'left_smalltoe', 'left_heel', 'right_bigtoe',
    'right_smalltoe', 'right_heel', 'neck_extra', 'pelvis_extra',
    'thorax_extra', 'spine_extra', 'neck_openpose', 'right_elbow_openpose',
    'right_wrist_openpose', 'left_elbow_openpose', 'left_wrist_openpose',
    'pelvis_openpose', 'right_knee_openpose', 'right_ankle_openpose',
    'left_knee_openpose', 'left_ankle_openpose', 'left_bigtoe_openpose',
    'left_smalltoe_openpose', 'left_heel_openpose', 'right_bigtoe_openpose',
    'right_smalltoe_openpose', 'right_heel_openpose', 'spine_4_3dhp',
    'left_clavicle_3dhp', 'right_clavicle_3dhp', 'left_toe_3dhp',
    'right_toe_3dhp'
]

HUMAN_DATA_PARTS = {
    'head': HUMAN_DATA_HEAD,
    'left_hand': HUMAN_DATA_LEFT_HAND,
    'right_hand': HUMAN_DATA_RIGHT_HAND,
    'shoulder': HUMAN_DATA_SHOULDER,
    'hip': HUMAN_DATA_HIP,
    'body': HUMAN_DATA_BODY
}

HUMAN_DATA_LIMBS = {
    'body': [
        ['pelvis', 'left_hip'],
        ['pelvis', 'right_hip'],
        ['pelvis', 'spine_1'],
        ['spine_1', 'spine_2'],
        ['spine_2', 'spine_3'],
        ['spine_3', 'neck'],
        ['neck', 'head'],
        ['left_ankle', 'left_knee'],
        ['left_knee', 'left_hip'],
        ['right_ankle', 'right_knee'],
        ['right_knee', 'right_hip'],
        ['right_ankle', 'right_foot'],
        ['left_ankle', 'left_foot'],
        ['left_hip', 'right_hip'],
        ['left_shoulder', 'left_hip'],
        ['right_shoulder', 'right_hip'],
        ['left_collar', 'spine_3'],
        ['right_collar', 'spine_3'],
        ['right_collar', 'right_shoulder'],
        ['left_collar', 'left_shoulder'],
        ['left_shoulder', 'right_shoulder'],
        ['left_shoulder', 'left_elbow'],
        ['right_shoulder', 'right_elbow'],
        ['left_elbow', 'left_wrist'],
        ['right_elbow', 'right_wrist'],
        ['left_ankle', 'left_bigtoe'],
        ['left_ankle', 'left_smalltoe'],
        ['left_ankle', 'left_heel'],
        ['right_ankle', 'right_bigtoe'],
        ['right_ankle', 'right_smalltoe'],
        ['right_ankle', 'right_heel'],
        ['left_shoulder', 'left_ear'],
        ['right_shoulder', 'right_ear'],
        ['right_ear', 'right_eye'],
        ['right_eye', 'nose'],
        ['nose', 'left_eye'],
        ['left_eye', 'left_ear'],
        ['nose', 'jaw'],
        ['jaw', 'neck'],
        # extra limbs
        ['pelvis_extra', 'left_hip_extra'],
        ['pelvis_extra', 'right_hip_extra'],
        ['left_hip_extra', 'left_knee'],
        ['right_hip_extra', 'right_knee'],
        ['left_hip_extra', 'left_shoulder'],
        ['right_hip_extra', 'right_shoulder'],
        ['pelvis_extra', 'spine_1'],
        ['spine_2', 'spine_extra'],
        ['spine_extra', 'spine_3'],
        ['spine_3', 'thorax_extra'],
        ['thorax_extra', 'left_shoulder'],
        ['thorax_extra', 'right_shoulder'],
        ['thorax_extra', 'neck_extra'],
        ['neck_extra', 'jaw_extra'],
        ['jaw_extra', 'nose'],
        ['head_extra', 'nose'],
        ['head_extra', 'headtop'],
        ['head_extra', 'neck_extra'],
        ['neck_extra', 'headtop'],
        ['right_hip_extra', 'left_hip_extra'],
        ['right_eye_openpose', 'right_ear_openpose'],
        ['left_ear_openpose', 'left_eye_openpose'],
        ['right_shoulder_openpose', 'right_elbow_openpose'],
        ['right_elbow_openpose', 'right_wrist_openpose'],
        ['left_shoulder_openpose', 'right_shoulder_openpose'],
        ['left_shoulder_openpose', 'left_elbow_openpose'],
        ['left_elbow_openpose', 'left_wrist_openpose'],
        ['pelvis_openpose', 'headtop'],
        ['pelvis_openpose', 'headtop'],
        ['neck_extra', 'right_hip_openpose'],
        ['neck_extra', 'left_hip_openpose'],
        ['right_hip_openpose', 'right_shoulder_openpose'],
        ['right_hip_openpose', 'right_knee_openpose'],
        ['left_hip_openpose', 'left_shoulder_openpose'],
        ['left_hip_openpose', 'left_knee_openpose'],
        ['right_knee_openpose', 'right_ankle_openpose'],
        ['left_knee_openpose', 'left_ankle_openpose'],
        ['right_ankle_openpose', 'right_heel_openpose'],
        ['left_ankle_openpose', 'left_heel_openpose'],
        ['right_heel_openpose', 'right_bigtoe_openpose'],
        ['right_heel_openpose', 'right_smalltoe_openpose'],
        ['left_ankle_openpose', 'left_bigtoe_openpose'],
        ['left_ankle_openpose', 'left_smalltoe_openpose'],
    ],
    'face': [['face_contour_1', 'face_contour_2'],
             ['face_contour_2', 'face_contour_3'],
             ['face_contour_3', 'face_contour_4'],
             ['face_contour_4', 'face_contour_5'],
             ['face_contour_5', 'face_contour_6'],
             ['face_contour_6', 'face_contour_7'],
             ['face_contour_7', 'face_contour_8'],
             ['face_contour_8', 'face_contour_9'],
             ['face_contour_9', 'face_contour_10'],
             ['face_contour_10', 'face_contour_11'],
             ['face_contour_11', 'face_contour_12'],
             ['face_contour_12', 'face_contour_13'],
             ['face_contour_13', 'face_contour_14'],
             ['face_contour_14', 'face_contour_15'],
             ['face_contour_15', 'face_contour_16'],
             ['face_contour_16', 'face_contour_17']],
    'left_hand':
    [['left_wrist', 'left_thumb_1'], ['left_thumb_1', 'left_thumb_2'],
     ['left_thumb_2', 'left_thumb_3'], ['left_thumb_3', 'left_thumb'],
     ['left_wrist', 'left_index_1'], ['left_index_1', 'left_index_2'],
     ['left_index_2', 'left_index_3'], ['left_index_3', 'left_index'],
     ['left_wrist', 'left_middle_1'], ['left_middle_1', 'left_middle_2'],
     ['left_middle_2', 'left_middle_3'], ['left_middle_3', 'left_middle'],
     ['left_wrist', 'left_ring_1'], ['left_ring_1', 'left_ring_2'],
     ['left_ring_2', 'left_ring_3'], ['left_ring_3', 'left_ring'],
     ['left_wrist', 'left_pinky_1'], ['left_pinky_1', 'left_pinky_2'],
     ['left_pinky_2', 'left_pinky_3'], ['left_pinky_3', 'left_pinky']],
    'right_hand': [['right_wrist', 'right_thumb_1'],
                   ['right_thumb_1', 'right_thumb_2'],
                   ['right_thumb_2', 'right_thumb_3'],
                   ['right_thumb_3', 'right_thumb'],
                   ['right_wrist', 'right_index_1'],
                   ['right_index_1', 'right_index_2'],
                   ['right_index_2', 'right_index_3'],
                   ['right_index_3', 'right_index'],
                   ['right_wrist', 'right_middle_1'],
                   ['right_middle_1', 'right_middle_2'],
                   ['right_middle_2', 'right_middle_3'],
                   ['right_middle_3', 'right_middle'],
                   ['right_wrist', 'right_ring_1'],
                   ['right_ring_1', 'right_ring_2'],
                   ['right_ring_2', 'right_ring_3'],
                   ['right_ring_3', 'right_ring'],
                   ['right_wrist', 'right_pinky_1'],
                   ['right_pinky_1', 'right_pinky_2'],
                   ['right_pinky_2', 'right_pinky_3'],
                   ['right_pinky_3', 'right_pinky']],
    'right_eye':
    [['right_eye_1', 'right_eye_2'], ['right_eye_2', 'right_eye_3'],
     ['right_eye_3', 'right_eye_4'], ['right_eye_4', 'right_eye_5'],
     ['right_eye_5', 'right_eye_6'], ['right_eye_6', 'right_eye_1'],
     ['right_eyebrow_1', 'right_eyebrow_2'],
     ['right_eyebrow_2', 'right_eyebrow_3'],
     ['right_eyebrow_3', 'right_eyebrow_4'],
     ['right_eyebrow_4', 'right_eyebrow_5']],
    'left_eye': [['left_eye_4', 'left_eye_3'], ['left_eye_3', 'left_eye_2'],
                 ['left_eye_2', 'left_eye_1'], ['left_eye_1', 'left_eye_6'],
                 ['left_eye_6', 'left_eye_5'], ['left_eye_5', 'left_eye_4'],
                 ['left_eyebrow_1', 'left_eyebrow_2'],
                 ['left_eyebrow_2', 'left_eyebrow_3'],
                 ['left_eyebrow_3', 'left_eyebrow_4'],
                 ['left_eyebrow_4', 'left_eyebrow_5']],
    'mouth': [['mouth_1', 'mouth_2'], ['mouth_2', 'mouth_3'],
              ['mouth_3', 'mouth_4'], ['mouth_4', 'mouth_5'],
              ['mouth_5', 'mouth_6'], ['mouth_6', 'mouth_7'],
              ['mouth_7', 'mouth_8'], ['mouth_8', 'mouth_9'],
              ['mouth_9', 'mouth_10'], ['mouth_10', 'mouth_11'],
              ['mouth_11', 'mouth_12'], ['mouth_12', 'mouth_1'],
              ['lip_1', 'lip_2'], ['lip_2', 'lip_3'], ['lip_3', 'lip_4'],
              ['lip_4', 'lip_5'], ['lip_5', 'lip_6'], ['lip_6', 'lip_7'],
              ['lip_7', 'lip_8'], ['lip_8', 'lip_1']],
    'nose': [
        ['nosebridge_1', 'nosebridge_2'],
        ['nosebridge_2', 'nosebridge_3'],
        ['nosebridge_3', 'nosebridge_4'],
        ['nose_1', 'nose_2'],
        ['nose_2', 'nose_3'],
        ['nose_3', 'nose_4'],
        ['nose_4', 'nose_5'],
    ]
}

HUMAN_DATA_LIMBS_INDEX = {}
for k in HUMAN_DATA_LIMBS:
    HUMAN_DATA_LIMBS_INDEX[k] = [[
        HUMAN_DATA.index(limb[0]),
        HUMAN_DATA.index(limb[1])
    ] for limb in HUMAN_DATA_LIMBS[k]]

HUMAN_DATA_PALETTE = {
    'left_eye': [[0, 0, 0]],
    'right_eye': [[0, 0, 0]],
    'nose': [[0, 0, 255]],
    'mouth': [[0, 255, 255]],
    'face': [[255, 0, 0]],
    'left_hand': [[0, 0, 0]],
    'right_hand': [[0, 0, 0]],
}
