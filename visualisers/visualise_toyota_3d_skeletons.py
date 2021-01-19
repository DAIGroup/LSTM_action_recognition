""" LCR-Net: Localization-Classification-Regression for Human Pose
Copyright (C) 2017 Gregory Rogez & Philippe Weinzaepfel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>"""

import sys, os, pdb
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import cv2
import json

plt.ion()

video_shape = (480,640,3)

dataset_skel_path = '/media/pau/extern_wd/Datasets/ToyotaSmartHome/json'
skeleton_files = glob('%s/*.json' % dataset_skel_path)


def display_poses(image, detections, njts, fig):
    if njts == 13:
        left = [(9, 11), (7, 9), (1, 3), (3, 5)]  # bones on the left
        right = [(0, 2), (2, 4), (8, 10), (6, 8)]  # bones on the right
        right += [(4, 5), (10, 11)]  # bones on the torso
        # (manually add bone between middle of 4,5 to middle of 10,11, and middle of 10,11 and 12)
        head = 12
    elif njts == 17:
        left = [(9, 11), (7, 9), (1, 3), (3, 5)]  # bones on the left
        right = [(0, 2), (2, 4), (8, 10), (6, 8)]  # bones on the right and the center
        right += [(4, 13), (5, 13), (13, 14), (14, 15), (15, 16), (12, 16), (10, 15), (11, 15)]  # bones on the torso
        head = 16

    # 2D 
    ax = fig.add_subplot(211)
    ax.imshow(image)
    for det in detections:
        pose2d = det['pose2d']
        score = det['cumscore']
        lw = 2
        # draw green lines on the left side
        for i, j in left:
            ax.plot([pose2d[i], pose2d[j]], [pose2d[i + njts], pose2d[j + njts]], 'g', scalex=None, scaley=None, lw=lw)
        # draw blue linse on the right side and center
        for i, j in right:
            ax.plot([pose2d[i], pose2d[j]], [pose2d[i + njts], pose2d[j + njts]], 'b', scalex=None, scaley=None, lw=lw)
        if njts == 13:  # other bones on torso for 13 jts
            def avgpose2d(a, b, offset=0):  # return the coordinate of the middle of joint of index a and b
                return (pose2d[a + offset] + pose2d[b + offset]) / 2.0

            ax.plot([avgpose2d(4, 5), avgpose2d(10, 11)],
                    [avgpose2d(4, 5, offset=njts), avgpose2d(10, 11, offset=njts)], 'b', scalex=None, scaley=None,
                    lw=lw)
            ax.plot([avgpose2d(12, 12), avgpose2d(10, 11)],
                    [avgpose2d(12, 12, offset=njts), avgpose2d(10, 11, offset=njts)], 'b', scalex=None, scaley=None,
                    lw=lw)
            # put red markers for all joints
        for i in range(njts): ax.plot(pose2d[i], pose2d[i + njts], color='r', marker='.', scalex=None, scaley=None)
        # legend and ticks
        ax.text(pose2d[head] - 20, pose2d[head + njts] - 20, '%.1f' % (score), color='blue')
    ax.set_xticks([])
    ax.set_yticks([])

    # 3D
    ax = fig.add_subplot(212, projection='3d')
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
        # score
        ax.text(pose3d[head] + 0.1, pose3d[head + njts] + 0.1, pose3d[head + 2 * njts], '%.1f' % (score), color='blue')
    # legend and ticks
    # ax.set_aspect('equal')
    ax.elev = -90
    ax.azim = 90
    ax.dist = 8
    ax.set_xlabel('X axis', labelpad=-5)
    ax.set_ylabel('Y axis', labelpad=-5)
    ax.set_zlabel('Z axis', labelpad=-5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.show()
    # pdb.set_trace()


def main():
    fig = plt.figure()
    for i, skel_file in enumerate(skeleton_files[1:]):
        fh = open(skel_file, 'r')
        skel_data = json.load(fh)
        fh.close()

        skel_frames = skel_data['frames']
        num_joints = skel_data['njts']
        cv2.namedWindow('skeleton_view')

        video_file = skel_file.replace('json', 'mp4')
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait = 1000/fps
        for skel_frame in skel_frames:
            ret, img = cap.read()
            display_poses(img, skel_frame, num_joints, fig)
            cv2.imshow('skeleton_view', img)
            cv2.waitKey(int(wait/2))


if __name__ == '__main__':
    main()
