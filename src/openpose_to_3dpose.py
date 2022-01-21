import numpy as np
import tensorflow as tf
import data_utils
import re
import cameras
import json
import os
from predict_3dpose import create_model

FLAGS = tf.app.flags.FLAGS
keypoint_order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]


def read_json():
    json_dir = FLAGS.openpose_json_dir
    load_files = os.listdir(json_dir)
    json_files = sorted([file_name for file_name in load_files if file_name.endswith(".json")])

    for json_file in json_files:
        json_file = os.path.join(json_dir, json_file)
        data = json.load(open(json_file))
        content_data = data["people"][0]["pose_keypoints_2d"]
        xy = []
        for i in range(0, len(content_data), 3):
            xy.append(content_data[i])
            xy.append(content_data[i + 1])

        frame_index = re.findall("(\d+)", json_file)
        _xy = xy[0:19 * 2]
        for x in range(len(xy)):
            # del jnt 8
            if x == 8 * 2:
                del _xy[x]
            if x == 8 * 2 + 1:
                del _xy[x]
            # map jnt 9 to 8
            if x == 9 * 2:
                _xy[16] = xy[x]
                _xy[17] = xy[x + 1]
            # map jnt 10 to 9
            if x == 10 * 2:
                _xy[18] = xy[x]
                _xy[19] = xy[x + 1]
                # map jnt 11 to 10
            if x == 11 * 2:
                _xy[20] = xy[x]
                _xy[21] = xy[x + 1]
            # map jnt 12 to 11
            if x == 12 * 2:
                _xy[22] = xy[x]
                _xy[23] = xy[x + 1]
            # map jnt 13 to 12
            if x == 13 * 2:
                _xy[24] = xy[x]
                _xy[25] = xy[x + 1]
                # map jnt 14 to 13
            if x == 14 * 2:
                _xy[26] = xy[x]
                _xy[27] = xy[x + 1]
            # map jnt 15 to 14
            if x == 15 * 2:
                _xy[28] = xy[x]
                _xy[29] = xy[x + 1]
            # map jnt 16 to 15
            if x == 16 * 2:
                _xy[30] = xy[x]
                _xy[31] = xy[x + 1]
            # map jnt 17 to 16
            if x == 17 * 2:
                _xy[32] = xy[x]
                _xy[33] = xy[x + 1]
            # map jnt 18 to 17
            if x == 18 * 2:
                _xy[34] = xy[x]
                _xy[35] = xy[x + 1]
        # coco
        xy = _xy
        return xy


def predict_depth(keypoints):
    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]
    actions = data_utils.define_actions(FLAGS.action)

    # Load camera parameters
    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    this_file = os.path.dirname(os.path.realpath(__file__))
    rcams = cameras.load_cameras(os.path.join(this_file, "..", FLAGS.cameras_path), SUBJECT_IDS)

    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data(
        actions, FLAGS.data_dir, rcams, 2)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    device_count = {"GPU": 1}
    before_pose = None
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as session:
        batch_size = 128
        model = create_model(session, actions, batch_size)
        iter_range = len(keypoints.keys())
        export_units = {}
        for n, (frame, xy) in enumerate(keypoints.items()):
            print("Frame {0} / {1}".format(frame, iter_range))
            joints_array = np.zeros((1, 36))
            joints_array[0] = [0 for i in range(36)]
            for i in range(len(joints_array[0])):
                joints_array[0][i] = float(xy[i])
            _data = joints_array[0]
            for i in range(len(keypoint_order)):
                for j in range(2):
                    enc_in[0][keypoint_order[i] * 2 + j] = _data[i * 2 + j]
            for j in range(2):
                # Hip
                enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                # Neck/Nose
                enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
                # Thorax
                enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]
            # set spine
            spine_x = enc_in[0][24]
            spine_y = enc_in[0][25]

            enc_in = enc_in[:, dim_to_use_2d]
            mu = data_mean_2d[dim_to_use_2d]
            stddev = data_std_2d[dim_to_use_2d]
            enc_in = np.divide((enc_in - mu), stddev)

            dp = 1.0
            dec_out = np.zeros((1, 48))
            dec_out[0] = [0 for i in range(48)]
            _, _, poses3d = model.step(session, enc_in, dec_out, dp, isTraining=False)
            all_poses_3d = []
            enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
            poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
            all_poses_3d.append(poses3d)
            enc_in, poses3d = map(np.vstack, [enc_in, all_poses_3d])
            _max = 0
            _min = 10000

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    tmp = poses3d[i][j * 3 + 2]
                    poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                    poses3d[i][j * 3 + 1] = tmp
                    if poses3d[i][j * 3 + 2] > _max:
                        _max = poses3d[i][j * 3 + 2]
                    if poses3d[i][j * 3 + 2] < _min:
                        _min = poses3d[i][j * 3 + 2]

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min
                    poses3d[i][j * 3] += (spine_x - 630)
                    poses3d[i][j * 3 + 2] += (500 - spine_y)

            if np.min(poses3d) < -1000:
                poses3d = before_pose
            p3d = poses3d
            x, y, z = [[] for _ in range(3)]
            if not poses3d is None:
                to_export = poses3d.tolist()[0]
            else:
                to_export = [0.0 for _ in range(96)]
            for i in range(0, len(to_export), 3):
                x.append(to_export[i])
                y.append(to_export[i + 1])
                z.append(to_export[i + 2])

            export_units[frame] = {}
            for jnt_index, (_x, _y, _z) in enumerate(zip(x, y, z)):
                export_units[frame][jnt_index] = {"translate": [_x, _y, _z]}

            before_pose = poses3d

    _out_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'maya/3d_data.json')
    with open(_out_file, 'w') as outfile:
        json.dump(export_units, outfile)
        print("exported json to {0}".format(_out_file))
        print("!!!!! DONE !!!!!")


def main():
    openpose_keypoints = read_json()
    predict_depth(openpose_keypoints)


if __name__ == "__main__":
    tf.app.run()
