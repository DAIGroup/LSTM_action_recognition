"""
    Script to load and visualise skeleton information from SmartHome dataset.
"""
from glob import glob
import json
import cv2
import numpy as np

video_shape = (480,640,3)

dataset_skel_path = '/media/pau/extern_wd/Datasets/ToyotaSmartHome/json'
skeleton_files = glob('%s/*.json' % dataset_skel_path)

bones = [(0, 2), (2, 4), (1, 3), (3, 5), (4, 5), (4, 11), (5, 10), (10, 11), (10, 8), (8, 6), (11, 9), (9, 7)]

colors = [(0,0,255), (0,128,255), (0,255,255), (0,255,0), (128,255,0), (255,0,0), (255,0,255)]

def draw_2d_skeleton(img, skel_2d, index):
    for j, joint_2d in enumerate(skel_2d):
        x, y = int(joint_2d[0]), int(joint_2d[1])
        if j == 12:  # We draw a circle for the head
            cv2.circle(img, (x,y), radius=15, color=colors[index], thickness=1)
        else:
            cv2.rectangle(img, (x, y), (x, y), color=colors[index], thickness=3)

        cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_PLAIN, fontScale=.6, thickness=1, color=(255, 255, 0))
    for bone in bones:
        j1, j2 = bone
        x1, y1 = int(skel_2d[j1, 0]), int(skel_2d[j1, 1])
        x2, y2 = int(skel_2d[j2, 0]), int(skel_2d[j2, 1])
        cv2.line(img, (x1, y1), (x2, y2), color=colors[index])


def main():
    for i, skel_file in enumerate(skeleton_files):
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
            for s, skeleton in enumerate(skel_frame):
                skel_2d = np.array(skeleton['pose2d']).reshape((2, num_joints)).T
                draw_2d_skeleton(img, skel_2d, s)
            cv2.imshow('skeleton_view', img)
            cv2.waitKey(int(wait/2))


if __name__ == '__main__':
    main()



