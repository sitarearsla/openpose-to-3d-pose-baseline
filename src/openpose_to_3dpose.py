import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import data_utils
import viz
import re
import cameras
import json
import os
from predict_3dpose import create_model
import imageio
import logging
from scipy.interpolate import UnivariateSpline

FLAGS = tf.app.flags.FLAGS


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
    SUBJECT_IDS = [5, 6, 7, 8, 9, 11]
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
            joints_array = np.zeros((1, 36))
            joints_array[0] = [0 for i in range(36)]


def main():
    openpose_json = read_json()
    predict_depth(openpose_json)


if __name__ == "__main__":
    tf.app.run()
