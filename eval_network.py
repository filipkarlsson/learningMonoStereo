# - Train -

import os
import sys
import numpy as np
from PIL import Image
import h5py
import pickle

import json
import tensorflow as tf
from matplotlib import pyplot as plt

from skimage.measure import block_reduce

import bootstrap
import vis
import custom_losses

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam


import matplotlib.pyplot as plt
from datetime import datetime

my = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(my, '..', 'demon', 'python'))


from depthmotionnet.datareader import *


TRAINING_SAVE_FOLDER = 'training_3/'


def load_batch():
    data_tensors_keys = ('IMAGE_PAIR', 'MOTION', 'FLOW',
                         'DEPTH', 'INTRINSICS', 'DEPTHMASKS')

    # the following parameters are just an example and are not optimized for
    # training
    reader_params = {
        'batch_size': 66,
        'test_phase': False,
        'builder_threads': 4,
        'inverse_depth': True,
        'motion_format': 'ANGLEAXIS6',
        'norm_trans_scale_depth': True,
        # downsampling of image and depth is supported 192 256
        'scaled_height': 192,
        'scaled_width': 256,
        'scene_pool_size': 5,  # for actual training this should be around 500
        'augment_rot180': 0,
        'augment_mirror_x': 0,
        'top_output': data_tensors_keys,  # request data tensors
        #'source': [{'path': 'train_data/traindata/sun3d_train_0.01m_to_0.1m.h5;train_data/traindata/sun3d_train_0.1m_to_0.2m.h5;train_data/traindata/sun3d_train_0.4m_to_0.8m.h5;train_data/traindata/sun3d_train_0.8m_to_1.6m.h5;train_data/traindata/sun3d_train_1.6m_to_infm.h5', 'weight': [{'t': 0, 'v': 1.0}]}, ],
        'source': [
            {'path': 'test_data/mvs_test.h5', 'weight': [{'t': 0, 'v': 1.0}]},
            #{'path': 'test_data/nyu2_test.h5', 'weight': [{'t': 0, 'v': 1.0}]},
            #{'path': 'test_data/rgbd_test.h5', 'weight': [{'t': 0, 'v': 1.0}]},
            #{'path': 'test_data/scenes11_test.h5', 'weight': [{'t': 0, 'v': 1.0}]},
            #{'path': 'test_data/sun3d_test.h5', 'weight': [{'t': 0, 'v': 1.0}]},
        ],
    }

    reader_tensors = multi_vi_h5_data_reader(
        len(data_tensors_keys), json.dumps(reader_params))
    info = reader_tensors[0]
    sample_id = reader_tensors[1]

    # create a dict to make the distinct data tensors accessible via keys
    data_dict = dict(zip(data_tensors_keys, reader_tensors[2]))

    gpu_options = tf.GPUOptions()
    # leave some memory to other processes
    gpu_options.per_process_gpu_memory_fraction = 0.8

    session = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, gpu_options=gpu_options))

    while True:
        yield session.run(data_dict)


def generate_train_data(data_loader):
    for mini_batch in data_loader:

        image_pair_full = mini_batch['IMAGE_PAIR']

        flow = mini_batch['FLOW']

        # Normalize
        flow[:, 0, :, :] *= 1 / 256.0
        flow[:, 1, :, :] *= 1 / 192.0

        flow_64_48 = block_reduce(flow, (1, 1, 4, 4), func=np.nanmean)
        # print(flow_64_48)
        # flow_64_48 = np.nan_to_num(flow_64_48)
        flow_8_6 = block_reduce(flow_64_48, (1, 1, 8, 8),  func=np.nanmean)

        image_pair_full = np.moveaxis(image_pair_full, 1, -1)
        flow_64_48 = np.moveaxis(flow_64_48, 1, -1)
        flow_8_6 = np.moveaxis(flow_8_6, 1, -1)

        yield (image_pair_full, [flow_8_6, flow_8_6, flow_64_48, flow_64_48])


def eval_net(data_loader):
    import matplotlib.image as mpimg

    checkpoint_path = TRAINING_SAVE_FOLDER + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    bootstrap_net = bootstrap.bootstrap_net()

    bootstrap_net.load_weights(checkpoint_path)
    adam = Adam(decay=0.0004)

    loss_list = [custom_losses.flow_loss, custom_losses.flow_conf_loss_from_flow(pred_flow=bootstrap_net.outputs[0]),
                     custom_losses.flow_loss,  custom_losses.flow_conf_loss_from_flow(pred_flow=bootstrap_net.outputs[2])]

    bootstrap_net.compile(loss=loss_list, optimizer=adam)

    test_set = next(generate_train_data(data_loader))


    test_loss = bootstrap_net.evaluate_generator(
        generate_train_data(data_loader), steps=1)


if __name__ == '__main__':
    data_loader = load_batch()
    #train_bootstrap(data_loader)
    eval_net(data_loader)

    tf.keras.backend.clear_session()
