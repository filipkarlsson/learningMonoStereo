# - Train -

import os
import sys
import numpy as np
from PIL import Image
import h5py
import pickle

import json
import tensorflow as tf
from tensorflow.python import keras

from matplotlib import pyplot as plt

from skimage.measure import block_reduce

import bootstrap
import iterative_network
import vis
import custom_losses

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Model



import matplotlib.pyplot as plt
from datetime import datetime

my = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(my, '..', 'demon', 'python'))
sys.path.insert(0, os.path.join(my, '../demon/lmbspecialops/python'))

from depthmotionnet.datareader import *
import lmbspecialops

TRAINING_SAVE_FOLDER = 'training_4/'


def load_batch():
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    data_tensors_keys = ('IMAGE_PAIR', 'MOTION', 'FLOW',
                         'DEPTH', 'INTRINSICS', 'DEPTHMASKS')

    reader_params = {
        'batch_size': 32,
        'test_phase': False,
        'builder_threads': 4,
        'inverse_depth': True,
        'motion_format': 'ANGLEAXIS6',
        'norm_trans_scale_depth': True,
        # downsampling of image and depth is supported 192 256
        'scaled_height': 192,
        'scaled_width': 256,
        'scene_pool_size': 5,  # for actual training this should be around 500
        'augment_rot180': 0.5,
        'augment_mirror_x': 0.5,
        'top_output': data_tensors_keys,  # request data tensors
        #'source': [{'path': 'train_data/traindata/sun3d_train_0.01m_to_0.1m.h5;train_data/traindata/sun3d_train_0.1m_to_0.2m.h5;train_data/traindata/sun3d_train_0.4m_to_0.8m.h5;train_data/traindata/sun3d_train_0.8m_to_1.6m.h5;train_data/traindata/sun3d_train_1.6m_to_infm.h5', 'weight': [{'t': 0, 'v': 1.0}]}, ],
        'source': [
            {'path': 'train_data/traindata/sun3d_train_0.01m_to_0.1m.h5', 'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/sun3d_train_0.1m_to_0.2m.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/sun3d_train_0.2m_to_0.4m.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/sun3d_train_0.4m_to_0.8m.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/sun3d_train_0.8m_to_1.6m.h5',
                 'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/sun3d_train_1.6m_to_infm.h5',
                 'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/mvs_achteck_turm.h5',
                 'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/mvs_breisach.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/mvs_citywall.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/rgbd_10_to_20_handheld_train.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/rgbd_10_to_20_3d_train.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/rgbd_10_to_20_simple_train.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
            {'path': 'train_data/traindata/scenes11_train.h5',
                'weight': [{'t': 0, 'v': 1.0}]},
        ],
    }

    reader_tensors = multi_vi_h5_data_reader(
        len(data_tensors_keys), json.dumps(reader_params))
    info = reader_tensors[0]
    sample_id = reader_tensors[1]

    # create a dict to make the distinct data tensors accessible via keys
    data_dict = dict(zip(data_tensors_keys, reader_tensors[2]))

    while True:
        yield sess.run(data_dict)


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


def generate_motion_depth_normals(data_loader):

    for mini_batch in data_loader:
        image_pair_full = mini_batch['IMAGE_PAIR']
        image_pair_full = np.moveaxis(image_pair_full, 1, -1)
        depth = mini_batch['DEPTH']

        depth_64_48 = block_reduce(depth, (1, 1, 4, 4), func=np.nanmean)
        depth_64_48 = np.moveaxis(depth_64_48, 1, -1)

        motion = mini_batch['MOTION']

        yield (image_pair_full, [motion, depth_64_48, depth_64_48])
        # outputs=[motion_output, depth_output, normals_output])


def generate_motion_depth_normals_test(data_loader):

    for mini_batch in data_loader:
        image_pair_full = mini_batch['IMAGE_PAIR']
        image_pair_full = np.moveaxis(image_pair_full, 1, -1)
        depth = mini_batch['DEPTH']

        #print(depth.shape)

        depth_64_48 = block_reduce(depth, (1, 1, 4, 4), func=np.nanmean)

        #print(depth_64_48.shape)

        depth_64_48 = np.moveaxis(depth_64_48, 1, -1)
        #print(depth_64_48.shape)


        motion = mini_batch['MOTION']

        flow = mini_batch['FLOW']

        print(motion.shape)


        # Normalize
        flow[:, 0, :, :] *= 1 / 256.0
        flow[:, 1, :, :] *= 1 / 192.0

        flow_64_48 = block_reduce(flow, (1, 1, 4, 4), func=np.nanmean)
        flow_64_48 = np.moveaxis(flow_64_48, 1, -1)

        print(flow_64_48.shape)
        # flow_64_48 = np.nan_to_num(flow_64_48)

        yield ([image_pair_full, flow_64_48, flow_64_48], [motion, depth_64_48, depth_64_48])
        #inputs=[image_pair, optical_flow_input, optical_flow_conf_input],
         #                                 outputs=[motion_output, depth_output, normals_output])



def train_bootstrap(data_loader):
    checkpoint_path = TRAINING_SAVE_FOLDER + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=1000)
    # create terminate on nan callback
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    bootstrap_net = bootstrap.bootstrap_net(32)

    # Compile model
    adam = Adam(decay=0.0004)
    loss_list = [custom_losses.flow_loss, custom_losses.flow_conf_loss_from_flow(pred_flow=bootstrap_net.outputs[0]),
                 custom_losses.flow_loss,  custom_losses.flow_conf_loss_from_flow(pred_flow=bootstrap_net.outputs[2])]

    bootstrap_net.compile(loss=loss_list, optimizer=adam)


    # Train model for 10k epochs
    history = bootstrap_net.fit_generator(
        generate_train_data(data_loader), steps_per_epoch=1, epochs=10000, callbacks=[cp_callback, nan_callback])

    with open(TRAINING_SAVE_FOLDER + 'train_history_first_10k.pickle', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)


    # Recompile model with additional scale invariant gradient loss funciton
    loss_list = [custom_losses.flow_loss_with_scale_inv_loss_8_6, custom_losses.flow_conf_loss_from_flow(pred_flow=bootstrap_net.outputs[0]),
                 custom_losses.flow_loss_with_scale_inv_loss_64_48,  custom_losses.flow_conf_loss_from_flow(pred_flow=bootstrap_net.outputs[2])]

    bootstrap_net.compile(loss=loss_list, optimizer=adam)

    # Train for 240k epochs
    history = bootstrap_net.fit_generator(
        generate_train_data(data_loader), steps_per_epoch=1, epochs=240000, callbacks=[cp_callback, nan_callback])

    with open(TRAINING_SAVE_FOLDER + 'train_history.pickle', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)


def full_bootstrap(batch_size):
    bootstrap_flow = bootstrap.bootstrap_net(batch_size)

    bootstrap_flow._make_predict_function()


    bootstrap_out = bootstrap_flow.outputs

    image_pair = bootstrap_flow.inputs[0]
    flow_out = bootstrap_out[2]
    conf_out = bootstrap_out[3]

    bootstrap_motion_input = [image_pair, flow_out, conf_out]

    bootstrap_motion_outputs = bootstrap.bootstrap_net_depth(batch_size)(bootstrap_motion_input)

    bootstrap_full = Model(inputs=image_pair, outputs=bootstrap_motion_outputs)
    bootstrap_full._make_predict_function()

    return bootstrap_full



def train_bootstrap_depth_motion(data_loader):
    checkpoint_path = TRAINING_SAVE_FOLDER + "cp_depth_motion_bootstrap.ckpt"
    flow_checkpoint_path = TRAINING_SAVE_FOLDER + "cp.ckpt"

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=1000)
    # create terminate on nan callback
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    bootstrap_flow = bootstrap.bootstrap_net(32)

    # Load weights
    bootstrap_flow.load_weights(flow_checkpoint_path)
    bootstrap_flow._make_predict_function()

    # Fix weights
    for layer in bootstrap_flow.layers:
        layer.trainable = False

    print(bootstrap_flow.summary())

    bootstrap_out = bootstrap_flow.outputs

    image_pair = bootstrap_flow.inputs[0]
    flow_out = bootstrap_out[2]
    conf_out = bootstrap_out[3]

    bootstrap_motion_input = [image_pair, flow_out, conf_out]

    bootstrap_motion_outputs = bootstrap.bootstrap_net_depth(32)(bootstrap_motion_input)

    bootstrap_full = Model(inputs=image_pair, outputs=bootstrap_motion_outputs)
    bootstrap_full._make_predict_function()


    # Compile model
    adam = Adam(decay=0.0004)
    loss_list = [custom_losses.mean_abs_error,
                 custom_losses.euclidean_with_gradient_loss, custom_losses.normals_loss_from_depth_gt]
    weights = [15, 300, 100]
    bootstrap_full.compile(loss=loss_list, optimizer=adam,  loss_weights=weights)
    print(bootstrap_full.summary())

    # plot_model(bootstrap_net, to_file='bootstrap.png', show_shapes=True)

    # Train model for 250k epochs
    history = bootstrap_full.fit_generator(
        generate_motion_depth_normals(data_loader), steps_per_epoch=1, epochs=250000, callbacks=[cp_callback, nan_callback])


    print("Training complete!")

    with open(TRAINING_SAVE_FOLDER + 'train_history_bootstrap_depth_motion.pickle', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)
        print("Saved weights to file")



def test_depth_and_motion(data_loader):
    checkpoint_path = TRAINING_SAVE_FOLDER + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=1000)
    # create terminate on nan callback
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    bootstrap_motion = bootstrap.bootstrap_net_depth(32)

    # Compile model
    adam = Adam(decay=0.0004)
    loss_list = [custom_losses.mean_abs_error,
                 custom_losses.euclidean_with_gradient_loss, custom_losses.normals_loss_from_depth_gt]

    bootstrap_motion.compile(loss=loss_list, optimizer=adam)
    print(bootstrap_motion.summary())

    #plot_model(bootstrap_net, to_file='bootstrap.png', show_shapes=True)

    # Train model for 10k epochs
    history = bootstrap_motion.fit_generator(
        generate_motion_depth_normals_test(data_loader), steps_per_epoch=1, epochs=10, callbacks=[cp_callback, nan_callback])



def train_iterative_net(data_loader, batch_size):
    checkpoint_path_bootstrap = TRAINING_SAVE_FOLDER + "cp_depth_motion_bootstrap.ckpt"
    checkpoint_path = TRAINING_SAVE_FOLDER + "cp_iterative_flow.ckpt"

    # Load previous netwoks
    bootstrap_network = full_bootstrap(batch_size)

    # Load weights
    bootstrap_network.load_weights(checkpoint_path_bootstrap)

    # Fix weights
    for layer in bootstrap_network.layers:
        layer.trainable = False


    bootstrap_inputs = bootstrap_network.inputs
    bootstrap_outputs = bootstrap_network.outputs # Motion, depth, normals

    iterative_network_flow = iterative_network.iterative_net_flow(batch_size)

    iterative_net_flow_inputs = bootstrap_inputs + bootstrap_outputs

    iterative_network_flow_outputs = iterative_network_flow(iterative_net_flow_inputs)

    combined_network = Model(inputs=bootstrap_inputs, outputs=iterative_network_flow_outputs)
    print(combined_network.summary())

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=1000)
    # create terminate on nan callback
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    # Compile model
    adam = Adam(decay=0.0004)
    loss_list = [custom_losses.flow_loss, custom_losses.flow_conf_loss_from_flow(pred_flow=combined_network.outputs[0]),
                 custom_losses.flow_loss,  custom_losses.flow_conf_loss_from_flow(pred_flow=combined_network.outputs[2])]

    combined_network.compile(loss=loss_list, optimizer=adam)


    # Train model for 10k epochs
    history = combined_network.fit_generator(
        generate_train_data(data_loader), steps_per_epoch=1, epochs=10000, callbacks=[cp_callback, nan_callback])

    with open(TRAINING_SAVE_FOLDER + 'train_history_iterative_flow_first_10k.pickle', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)


    # Recompile model with additional scale invariant gradient loss funciton
    loss_list = [custom_losses.flow_loss_with_scale_inv_loss_8_6, custom_losses.flow_conf_loss_from_flow(pred_flow=combined_network.outputs[0]),
                 custom_losses.flow_loss_with_scale_inv_loss_64_48,  custom_losses.flow_conf_loss_from_flow(pred_flow=combined_network.outputs[2])]

    combined_network.compile(loss=loss_list, optimizer=adam)

    # Train for 240k epochs
    history = combined_network.fit_generator(
        generate_train_data(data_loader), steps_per_epoch=1, epochs=240000, callbacks=[cp_callback, nan_callback])

    with open(TRAINING_SAVE_FOLDER + 'train_history_iterative_flow.pickle', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)



def train_iterative_net_depth_and_motion(data_loader, batch_size):
    # Init prev network
    checkpoint_path_iter_flow = TRAINING_SAVE_FOLDER + "cp_iterative_flow.ckpt"
    checkpoint_path = TRAINING_SAVE_FOLDER + "cp_iterative_depth_motion.ckpt"

    # Load previous netwoks
    bootstrap_network = full_bootstrap(batch_size)

    bootstrap_inputs = bootstrap_network.inputs
    bootstrap_outputs = bootstrap_network.outputs # Motion, depth, normals

    iterative_network_flow = iterative_network.iterative_net_flow(batch_size)

    iterative_net_flow_inputs = bootstrap_inputs + bootstrap_outputs

    iterative_network_flow_outputs = iterative_network_flow(iterative_net_flow_inputs)

    prev_network = Model(inputs=bootstrap_inputs, outputs=iterative_network_flow_outputs)

    # Load weights
    prev_network.load_weights(checkpoint_path_iter_flow)

    # Fix weights
    for layer in prev_network.layers:
        layer.trainable = False

    prev_network_out = prev_network.outputs

    image_pair = prev_network.inputs[0]
    flow_out = prev_network_out[2]
    conf_out = prev_network_out[3]
    motion_out = bootstrap_outputs[0]

    iterative_network_depth_inputs = [image_pair, flow_out, conf_out, motion_out]

    iterative_network_depth_outputs = iterative_network.iterative_net_depth(32)(iterative_network_depth_inputs)

    iterative_network_full = Model(inputs=iterative_network_depth_inputs, outputs=iterative_network_depth_outputs)
    print(iterative_network_full.summary())

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=1000)
    # create terminate on nan callback
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    # Compile model
    adam = Adam(decay=0.0004)
    loss_list = [custom_losses.mean_abs_error,
                 custom_losses.euclidean_with_gradient_loss, custom_losses.normals_loss_from_depth_gt]
    weights = [100, 300, 100]
    iterative_network_full.compile(loss=loss_list, optimizer=adam,  loss_weights=weights)
    print(iterative_network_full.summary())

    # plot_model(bootstrap_net, to_file='bootstrap.png', show_shapes=True)

    # Train model for 250k epochs
    history = iterative_network_full.fit_generator(
        generate_motion_depth_normals(data_loader), steps_per_epoch=1, epochs=250000, callbacks=[cp_callback, nan_callback])


    print("Training complete!")

    with open(TRAINING_SAVE_FOLDER + 'train_history_iterative_depth_motion.pickle', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)
        print("Saved weights to file")




def eval_net(data_loader):
    import matplotlib.image as mpimg

    checkpoint_path = TRAINING_SAVE_FOLDER + "cp_depth_motion_bootstrap.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    '''
    vis.show_train_history_from_pickle(
       TRAINING_SAVE_FOLDER + 'train_history_first_10k.pickle')

    vis.show_train_history_from_pickle(
        TRAINING_SAVE_FOLDER + 'train_history.pickle')
    '''
    #bootstrap_net = bootstrap.bootstrap_net()
    bootstrap_net = full_bootstrap(1)
    print("Created bootstrap network")

    bootstrap_net.load_weights(checkpoint_path)
    print("Weights loaded")

    test_set = next(generate_motion_depth_normals(data_loader))
    print("Loaded test data")

    test_img = test_set[0]
    gt_motion = test_set[1][0]
    gt_depth = test_set[1][1]

    img1 = mpimg.imread("gopher/im1_gopher.jpg")
    img2 = mpimg.imread("gopher/im2_gopher.jpg")
    #test_img = np.concatenate((img1, img2), axis=2).reshape(1, 192, 256, 6)

    print(test_img.shape)

    init = tf.global_variables_initializer().run()
    pred = bootstrap_net.predict(test_img)
    print("Prediction made")

    pred_motion = pred[0]
    pred_depth = pred[1]
    pred_normals = pred[2]

    print(pred_normals.shape)
    print(pred_depth.shape)
    print(pred_motion.shape)
    vis.vis_image_pair(test_img[0, :, :, :], "Input ")

    vis.vis_depth(pred_depth, gt_depth, "Depth")

    '''
    flow_8_6_gt = test_set[1][0]
    flow_64_48_gt = test_set[1][2]
    flow_8_6_pred = pred[0]
    flow_64_48_pred = pred[2]

    conf_8_6_pred = pred[1]
    conf_64_48_pred = pred[3]


    vis.vis_image_pair(test_img[0, :, :, :], "Input ")

    vis.vis_flow(flow_64_48_gt[0, :, :], (48, 64, 3), "64")
    vis.vis_flow(flow_64_48_pred[0, :, :], (48, 64, 3), "64_pred")

    vis.vis_flow(flow_8_6_gt[0, :, :], (6, 8, 3), "8")
    vis.vis_flow(flow_8_6_pred[0, :, :], (6, 8, 3), "8_pred")

    print(conf_64_48_pred.shape)
    print(conf_64_48_pred[0, :, :].shape)

    vis.vis_conf(conf_64_48_pred[0, :, :], "OF prdf conf 64x48")
    vis.vis_conf(conf_8_6_pred[0, :, :], "OF prdf conf 8x6")

    '''


if __name__ == '__main__':
    gpu_options = tf.GPUOptions()
    # leave some memory to other processes
    gpu_options.per_process_gpu_memory_fraction = 0.8

    #keras.backend.set_session()

    data_loader = load_batch()
    #train_bootstrap(data_loader)
    #eval_net(data_loader)
    #train_bootstrap_depth_motion(data_loader)
    #test_depth_and_motion(data_loader)
    train_iterative_net(data_loader, 32)

    tf.keras.backend.clear_session()
