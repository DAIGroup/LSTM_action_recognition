"""
    This script checks whether all skeletons are facing the camera after rotation.
"""
import json
from glob import glob
from preprocessing.generate_normalised_skeletons import joint_index, get_joint

dataset_skel_path = '/media/pau/extern_wd/Datasets/ToyotaSmartHome/json_rot'
skeleton_files = glob('%s/*.json' % dataset_skel_path)

def main():
    right_total = 0
    wrong_total = 0
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

        right = 0
        wrong = 0
        for skel_frame in skel_frames:
            if len(skel_frame) > 0:
                pose = skel_frame[0]['pose3d']

                for j in range(10,12,2):
                    lx, ly, lz = get_joint(pose, num_joints, j)
                    rx, ry, rz = get_joint(pose, num_joints, j+1)
                    if lx >= rx:
                        right += 1
                    else:
                        wrong += 1

        pct = 100 * (right / (right+wrong+10e-9))
        print('    right (%%): %.2f' % pct)
        right_total += right
        wrong_total += wrong

    final_pct = 100 * (right_total / (right_total + wrong_total + 10e-9))
    print("FINAL PERCENTAGE (RIGHT): %.2f" % final_pct)

if __name__ == '__main__':
    main()
