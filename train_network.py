# - Train -

import os
import sys
import numpy as np
from PIL import Image
import h5py

my = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(my, '..', 'demon-master', 'python'))


from depthmotionnet.datareader import *
import json
import tensorflow as tf
from matplotlib import pyplot as plt

from skimage.measure import block_reduce


import bootstrap
import vis

from keras.utils import plot_model


def load_batch():
    data_tensors_keys = ('IMAGE_PAIR', 'MOTION', 'FLOW',
                         'DEPTH', 'INTRINSICS', 'DEPTHMASKS')

    # the following parameters are just an example and are not optimized for
    # training
    reader_params = {
        'batch_size': 1,
        'test_phase': False,
        'builder_threads': 1,
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
        'source': [{'path': '../train_data/traindata/sun3d_train_0.01m_to_0.1m.h5', 'weight': [{'t': 0, 'v': 1.0}]}, ],
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

    # session = tf.InteractiveSession(config=tf.ConfigProto(
    #   allow_soft_placement=True, gpu_options=gpu_options))

    session = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, gpu_options=gpu_options))

    while True:
        yield session.run(data_dict)


def generate_train_data(data_loader):
    for mini_batch in data_loader:

        image_pair_full = mini_batch['IMAGE_PAIR']

        flow = mini_batch['FLOW']
        flow_64_48 = block_reduce(flow, (1, 1, 4, 4), func=np.nanmean)
        # print(flow_64_48)
        # flow_64_48 = np.nan_to_num(flow_64_48)
        flow_8_6 = block_reduce(flow_64_48, (1, 1, 8, 8),  func=np.nanmean)

        image_pair_full = np.moveaxis(image_pair_full, 1, -1)
        flow_64_48 = np.moveaxis(flow_64_48, 1, -1)
        flow_8_6 = np.moveaxis(flow_8_6, 1, -1)

        yield (image_pair_full, [flow_8_6, flow_8_6, flow_64_48, flow_64_48])


def train_bootstrap(data_loader):
    bootstrap_net = bootstrap.bootstrap_net()
    plot_model(bootstrap_net, to_file='bootstrap.png', show_shapes=True)

    history = bootstrap_net.fit_generator(
        generate_train_data(data_loader), steps_per_epoch=10, epochs=10)

    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    test_set = next(generate_train_data(data_loader))
    test_img = test_set[0]
    print(test_img.shape)
    pred = bootstrap_net.predict(test_img)

    flow_8_6_gt = test_set[1][0]
    flow_64_48_gt = test_set[1][2]
    flow_8_6_pred = pred[0]
    flow_64_48_pred = pred[2]

    vis.vis_flow(flow_64_48_gt[0, :, :], (48, 64, 3), "64")
    vis.vis_flow(flow_64_48_pred[0, :, :], (48, 64, 3), "64_pred")

    vis.vis_flow(flow_8_6_gt[0, :, :], (6, 8, 3), "8")
    vis.vis_flow(flow_8_6_pred[0, :, :], (6, 8, 3), "8_pred")


if __name__ == '__main__':
    data_loader = load_batch()
    train_bootstrap(data_loader)
