"""
    Script to rotate and normalise JSON skeleton files.
"""

import sys, os, pdb
import numpy as np
from glob import glob
import cv2
import json
import math

debug = True
copy2d = False

if debug:
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

video_shape = (480,640,3)

dataset_skel_path = '/media/pau/extern_wd/Datasets/ToyotaSmartHome/json'
skeleton_files = glob('%s/*.json' % dataset_skel_path)

dest_skel_path = '/media/pau/extern_wd/Datasets/ToyotaSmartHome/json_rot_ds'

if not os.path.exists(dest_skel_path):
    os.makedirs(dest_skel_path)

joint_index = {'left_foot': 0, 'right_foot': 1,
               'left_knee': 2, 'right_knee': 3,
               'left_hip': 4, 'right_hip': 5,
               'left_hand': 6, 'right_hand': 7,
               'left_elbow': 8, 'right_elbow': 9,
               'left_shoulder': 10, 'right_shoulder': 11,
               'head': 12}


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def display_poses(pos, title, detections, njts, fig):
    left = [(9, 11), (7, 9), (1, 3), (3, 5)]  # bones on the left
    right = [(0, 2), (2, 4), (8, 10), (6, 8)]  # bones on the right
    right += [(4, 5), (10, 11)]  # bones on the torso
    # (manually add bone between middle of 4,5 to middle of 10,11, and middle of 10,11 and 12)
    head = 12

    ax = fig.add_subplot(130+pos, projection='3d')
    ax.title.set_text(title)
    for i, det in enumerate(detections):
        pose3d = det['pose3d']
        score = det['cumscore']
        lw = 2

        def get_pair(i, j, offset):
            return [pose3d[i + offset], pose3d[j + offset]]

        def get_xyz_coord(i, j):
            return get_pair(i, j, 0), get_pair(i, j, njts), get_pair(i, j, njts * 2)

        # draw green lines on the left side
        for i, j in left:
            x, y, z = get_xyz_coord(i, j)
            ax.plot(x, y, z, 'g', scalex=None, scaley=None, lw=lw)
        # draw blue linse on the right side and center
        for i, j in right:
            x, y, z = get_xyz_coord(i, j)
            ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)
        if njts == 13:  # other bones on torso for 13 jts
            def avgpose3d(a, b, offset=0):
                return (pose3d[a + offset] + pose3d[b + offset]) / 2.0

            def get_avgpair(i1, i2, j1, j2, offset):
                return [avgpose3d(i1, i2, offset), avgpose3d(j1, j2, offset)]

            def get_xyz_avgcoord(i1, i2, j1, j2):
                return get_avgpair(i1, i2, j1, j2, 0), get_avgpair(i1, i2, j1, j2, njts), get_avgpair(i1, i2, j1, j2,
                                                                                                      njts * 2)

            x, y, z = get_xyz_avgcoord(4, 5, 10, 11)
            ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)
            x, y, z = get_xyz_avgcoord(12, 12, 10, 11)
            ax.plot(x, y, z, 'b', scalex=None, scaley=None, lw=lw)
        # put red markers for all joints
        for i in range(njts): ax.plot([pose3d[i]], [pose3d[i + njts]], [pose3d[i + njts * 2]], color='r', marker='.',
                                      scalex=None, scaley=None)
        ax.plot([pose3d[1]], [pose3d[1 + njts]], [pose3d[1 + njts * 2]], color='k', marker='*',
                scalex=None, scaley=None)
        ax.scatter([0],[0],[0], edgecolors='r', s=80, facecolors='w')
        # score
        ax.text(pose3d[head] + 0.1, pose3d[head + njts] + 0.1, pose3d[head + 2 * njts], '%.1f' % (score), color='blue')
    # legend and ticks
    # ax.set_aspect('equal')
    axis_equal_3d(ax)
    ax.elev = -90
    ax.azim = 90
    ax.dist = 8
    ax.set_xlabel('X', labelpad=-5)
    ax.set_ylabel('Y', labelpad=-5)
    ax.set_zlabel('Z', labelpad=-5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.show()
    # pdb.set_trace()


def get_joint(sk, num_joints, index):
    return sk[index], sk[num_joints + index], sk[2*num_joints + index]


def set_joint(sk, num_joints, index, x, y, z):
    sk[index] = x
    sk[num_joints + index] = y
    sk[2*num_joints + index] = z


def calculate_alpha_angle(skel, num_joints):
    """
    We call alpha de angle of rotation about the Y axis
    :param skel: skeleton joint information (list) xyz*num_joints long.
    :param num_joints: number of joints in the skeleton.
    :return: the angle about the Y axis with respect to the XY plane.
    """
    sl_x, _, sl_z = get_joint(skel, num_joints, joint_index['left_shoulder'])
    sr_x, _, sr_z = get_joint(skel, num_joints, joint_index['right_shoulder'])
    hr_x, _, hr_z = get_joint(skel, num_joints, joint_index['right_hip'])
    alpha = math.atan2((sl_z + sl_z)/2 - (sr_z + hr_z)/2, (sl_x + sl_x)/2 - (sr_x + hr_x)/2)
    return alpha


def calculate_beta_angle(skel, num_joints):
    """
    We call beta the angle of rotation about Z (rotation on the XY plane).
    :param skel: skeleton data consisting of 3*num_joints (xyz).
    :param num_joints: number of joints in the skeleton
    :return: the angle beta of rotation.
    """
    sl_x, sl_y, sl_z = get_joint(skel, num_joints, joint_index['left_shoulder'])
    sr_x, sr_y, sr_z = get_joint(skel, num_joints, joint_index['right_shoulder'])
    hl_x, hl_y, _ = get_joint(skel, num_joints, joint_index['left_hip'])
    hr_x, hr_y, _ = get_joint(skel, num_joints, joint_index['right_hip'])
    # beta = math.atan2((sl_y + sl_y) / 2 - (sr_y + hr_y) / 2, (sl_x + sl_x) / 2 - (sr_x + hr_x) / 2)
    beta_s = math.atan2(sl_y - sr_y, sl_x - sr_x)
    beta_h = math.atan2(hl_y - hr_y, hl_x - hr_x)
    beta = (beta_s + beta_h)/2
    # print('Angle btw. shoulders: %.2f' % (math.degrees(beta_s)))
    # print('Angle btw. hips     : %.2f' % math.degrees(beta_h))
    # print('Mean                : %.2f' % math.degrees(beta))
    return beta


def rotate_skeleton_beta(skel, num_joints, beta):
    rotated_skel = [0] * (num_joints * 3)

    for i in range(num_joints):
        pt_x, pt_y, pt_z = get_joint(skel, num_joints, i)
        a = pt_x #- nk_x
        b = pt_y #- nk_y
        npt_x = a * math.cos(beta) + b * math.sin(beta)
        npt_y = -a * math.sin(beta) + b * math.cos(beta)
        set_joint(rotated_skel, num_joints, i, npt_x, npt_y, pt_z)

    return rotated_skel


def rotate_skeleton_alpha(skel, num_joints, alpha):
    rotated_skel = [0] * (num_joints * 3)

    for i in range(num_joints):
        pt_x, pt_y, pt_z = get_joint(skel, num_joints, i)
        a = pt_x # - hd_x
        b = pt_z # - hd_z
        npt_x = a * math.cos(alpha) + b * math.sin(alpha)
        npt_z = -a * math.sin(alpha) + b * math.cos(alpha)
        set_joint(rotated_skel, num_joints, i, npt_x, pt_y, npt_z)

    return rotated_skel


def calculate_hand_head_distances(skel: list, num_joints: int):
    num_dists = 3
    new_skel = skel

    lh_x, lh_y, lh_z = get_joint(skel, num_joints, joint_index['left_hand'])
    rh_x, rh_y, rh_z = get_joint(skel, num_joints, joint_index['right_hand'])
    hd_x, hd_y, hd_z = get_joint(skel, num_joints, joint_index['head'])

    hand_to_hand = math.sqrt((lh_x - rh_x)**2 + (lh_y - rh_y)**2 + (lh_z - rh_z)**2)
    left_hand_to_head = math.sqrt((lh_x - hd_x)**2 + (lh_y - hd_y)**2 + (lh_z - hd_z)**2)
    right_hand_to_head = math.sqrt((rh_x - hd_x) ** 2 + (rh_y - hd_y) ** 2 + (rh_z - hd_z) ** 2)

    new_skel += [hand_to_hand, left_hand_to_head, right_hand_to_head]

    return new_skel



def rotate_skeletons_in_frame(skel_frame, num_joints, alphas, betas):
    rotated_skel_frame = skel_frame.copy()
    rotated_skel_frame_part = []
    rotated_skel_frame_part.append({})
    rotated_skel_frame_part[0]['cumscore'] = skel_frame[0]['cumscore']

    for i, skeleton in enumerate(skel_frame):
        rot_skel = rotate_skeleton_alpha(skeleton['pose3d'], num_joints, alphas[i])
        rotated_skel_frame_part[0]['pose3d'] = rot_skel.copy()
        beta = calculate_beta_angle(rot_skel, num_joints)
        rot_skel = rotate_skeleton_beta(rot_skel, num_joints, beta)
        # rot_skel_b_ds = calculate_hand_head_distances(rot_skel_b, num_joints)
        rotated_skel_frame[i]['pose3d'] = rot_skel
        if not copy2d:
            rotated_skel_frame[i]['pose2d'] = []

    return rotated_skel_frame, rotated_skel_frame_part


def main():
    if debug:
        fig = plt.figure()
    for i, skel_file in enumerate(skeleton_files):
        print('%5d/%5d: %s' % (i+1, len(skeleton_files), skel_file))
        fh = open(skel_file, 'r')
        skel_data = json.load(fh)
        fh.close()

        rot_skel_data = {}

        rot_skel_data['K'] = skel_data['K']
        rot_skel_data['njts'] = skel_data['njts']
        rot_skel_data['frames'] = []

        skel_frames = skel_data['frames']
        num_joints = skel_data['njts']
        # print(num_joints)
        if debug:
            cv2.namedWindow('skeleton_view')

        video_file = skel_file.replace('json', 'mp4')
        if debug:
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            wait = 1000/fps
        alphas = [0] * 5
        betas = [0] * 5
        s = 0
        for skel_frame in skel_frames:
            rot_skel_frame = []
            if len(skel_frame) > 0:
                if s == 0:
                    alpha = calculate_alpha_angle(skel_frame[0]['pose3d'], num_joints)
                    alphas[0] = alpha
                    # beta = calculate_beta_angle(skel_frame[0]['pose3d'], num_joints)
                    # betas[0] = beta
                if debug:
                    ret, img = cap.read()
                    display_poses(1, 'original', skel_frame, num_joints, fig)
                rot_skel_frame, rot_skel_frame_part = rotate_skeletons_in_frame(skel_frame, num_joints, alphas, betas)
                if debug:
                    display_poses(2, 'with $\\alpha$ rot. only', rot_skel_frame_part, num_joints, fig)
                    display_poses(3, 'final: with $\\beta$ rot.', rot_skel_frame, num_joints, fig)
                    cv2.imshow('skeleton_view', img)
                    # ch = cv2.waitKey(int(wait/2))
                    ch = cv2.waitKey(0)
                    if ch == 27:
                        sys.exit()
            rot_skel_data['frames'].append(rot_skel_frame)

        dst_skel_file = skel_file.replace(dataset_skel_path, dest_skel_path)
        if not debug:
            fh = open(dst_skel_file, 'w')
            json.dump(rot_skel_data, fh)
            fh.close()


if __name__ == '__main__':
    main()
