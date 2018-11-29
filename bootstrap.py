from tensorflow.python.keras.layers import Input, Dense, Conv2D, concatenate, Lambda, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import losses
from tensorflow import slice
import numpy as np

import custom_losses


def bootstrap_net():

    # 256x192x(3*2)
    image_pair = Input(shape=(192, 256, 6))

    layer_1a = Conv2D(32, kernel_size=(9, 1), strides=(
        2, 1), activation='relu', padding='same', input_shape=(192, 256, 6))(image_pair)
    layer_1b = Conv2D(32, kernel_size=(1, 9), strides=(
        1, 2), activation='relu', padding='same')(layer_1a)

    # 128x96x32
    layer_2a = Conv2D(64, kernel_size=(7, 1), strides=(
        2, 1), activation='relu', padding='same')(layer_1b)
    layer_2b = Conv2D(64, kernel_size=(1, 7), strides=(
        1, 2), activation='relu', padding='same')(layer_2a)

    layer3a = Conv2D(64, kernel_size=(3, 1), strides=(
        1, 1), activation='relu', padding='same')(layer_2b)
    # layer3b should be feeded forward later
    layer3b = Conv2D(64, kernel_size=(1, 3), strides=(
        1, 1), activation='relu', padding='same')(layer3a)

    layer4a = Conv2D(128, kernel_size=(5, 1), strides=(
        2, 1), activation='relu', padding='same')(layer3b)
    layer4b = Conv2D(128, kernel_size=(1, 5), strides=(
        1, 2), activation='relu', padding='same')(layer4a)

    layer5a = Conv2D(128, kernel_size=(3, 1), strides=(
        1, 1), activation='relu', padding='same')(layer4b)
    # Layer5b shall be feeded forward
    layer5b = Conv2D(128, kernel_size=(1, 3), strides=(
        1, 1), activation='relu', padding='same')(layer5a)

    layer6a = Conv2D(256, kernel_size=(5, 1), strides=(
        2, 1), activation='relu', padding='same')(layer5b)
    layer6b = Conv2D(256, kernel_size=(1, 5), strides=(
        1, 2), activation='relu', padding='same')(layer6a)

    # 16x12x256

    layer7a = Conv2D(256, kernel_size=(3, 1), strides=(
        1, 1), activation='relu', padding='same')(layer6b)
    # Layer7b shall be feeded forward
    layer7b = Conv2D(256, kernel_size=(1, 3), strides=(
        1, 1), activation='relu', padding='same')(layer7a)

    # 16x12x256

    layer8a = Conv2D(512, kernel_size=(3, 1), strides=(
        2, 1), activation='relu', padding='same')(layer7b)
    layer8b = Conv2D(512, kernel_size=(1, 3), strides=(
        1, 2), activation='relu', padding='same')(layer8a)

    layer9a = Conv2D(512, kernel_size=(3, 1), strides=(
        1, 1), activation='relu', padding='same')(layer8b)
    # Layer9b shall be feeded forward
    layer9b = Conv2D(512, kernel_size=(1, 3), strides=(
        1, 1), activation='relu', padding='same')(layer9a)

    # Enc of encoder part

    # First prediction branch
    layerFirstPred_1 = Conv2D(24, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', padding='same')(layer9b)

    layerFirstPred_2 = Conv2D(4, kernel_size=(3, 3), strides=(
        1, 1), activation=None, padding='same')(layerFirstPred_1)

    # Split tensor into optical flow (8x6x2) and into a confidense estimate
    # (8x6x2)
    first_pred_optical_flow = Lambda(lambda x: slice(
        x, (0, 0, 0, 0), (-1, -1, -1, 2)))(layerFirstPred_2)
    first_pred_confidence = Lambda(lambda x: slice(
        x, (0, 0, 0, 2), (-1, -1, -1, 2)))(layerFirstPred_2)

    merged_output = concatenate(
        [first_pred_optical_flow, first_pred_confidence])

    upconv1_first_pred = Conv2DTranspose(4, kernel_size=(
        4, 4), strides=(2, 2),  padding='same')(merged_output)

    # End of prediciton branch
    upconv1 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(
        2, 2),  padding='same', activation='relu')(layer9b)

    upconv1_merge = concatenate([upconv1_first_pred, upconv1, layer7b])

    upconv2 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(
        2, 2),  padding='same', activation='relu')(upconv1_merge)

    upconv2_merge = concatenate([upconv2, layer5b])

    upconv3 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(
        2, 2),  padding='same', activation='relu')(upconv2_merge)

    upconv3_merge = concatenate([upconv3, layer3b])

    layer10 = Conv2D(24, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', padding='same')(upconv3_merge)

    layer11 = Conv2D(4, kernel_size=(3, 3), strides=(
        1, 1), activation=None, padding='same')(layer10)

    # Output :D

    optical_flow_output = Lambda(lambda x: slice(
        x, (0, 0, 0, 0), (-1, -1, -1, 2)))(layer11)
    confidence_output = Lambda(lambda x: slice(
        x, (0, 0, 0, 2), (-1, -1, -1, 2)))(layer11)

    bootstrap_optical_flow = Model(inputs=image_pair, outputs=[
                                   first_pred_optical_flow, first_pred_confidence, optical_flow_output, confidence_output])

    adam = Adam(decay=0.0004)
    # bootstrap_optical_flow.compile(loss=losses.mean_squared_error,
    # optimizer=adam)

    # loss_list = [losses.mean_squared_error, custom_losses.flow_conf_loss_from_flow(pred_flow=first_pred_optical_flow),
    # losses.mean_squared_error,
    # custom_losses.flow_conf_loss_from_flow(pred_flow=confidence_output)]

    loss_list = [custom_losses.flow_loss, custom_losses.flow_conf_loss_from_flow(pred_flow=first_pred_optical_flow),
                 custom_losses.flow_loss,  custom_losses.flow_conf_loss_from_flow(pred_flow=confidence_output)]

    bootstrap_optical_flow.compile(
        loss=loss_list, optimizer=adam, loss_weights=[1.0, 200.0, 1.0, 200.0])

    return bootstrap_optical_flow


# - Train -
'''
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


data_tensors_keys = ('IMAGE_PAIR', 'MOTION', 'FLOW',
                     'DEPTH', 'INTRINSICS', 'DEPTHMASKS')

# the following parameters are just an example and are not optimized for
# training
reader_params = {
    'batch_size': 20,
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
# create a dict to make the distinct data tensors accessible via keys
data_dict = dict(zip(data_tensors_keys, reader_tensors[2]))

gpu_options = tf.GPUOptions()
# leave some memory to other processes
gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.InteractiveSession(config=tf.ConfigProto(
    allow_soft_placement=True, gpu_options=gpu_options))


batch = session.run(data_dict)
image_pair_full = batch['IMAGE_PAIR']
flow = batch['FLOW']

flow_64_48 = block_reduce(flow, (1, 1, 4, 4), func=np.nanmean)
# print(flow_64_48)
# flow_64_48 = np.nan_to_num(flow_64_48)
flow_8_6 = block_reduce(flow_64_48, (1, 1, 8, 8),  func=np.nanmean)


image_pair_full = np.moveaxis(image_pair_full, 1, -1)
flow_64_48 = np.moveaxis(flow_64_48, 1, -1)
flow_8_6 = np.moveaxis(flow_8_6, 1, -1)


print(str(np.sum(np.isnan(flow_8_6))) + "/" + str(np.size(flow_8_6)) +
      " = " + str(np.sum(np.isnan(flow_8_6)) / (np.size(flow_8_6))))


bootstrap_of = bootstrap_net()
bootstrap_of.fit(
    image_pair_full, [flow_8_6, flow_8_6, flow_64_48, flow_64_48], batch_size=1)

val_data = np.reshape(image_pair_full[0, :, :, :], (1, 192, 256, 6))
res = bootstrap_of.predict(
    val_data, batch_size=None, verbose=0, steps=None)

flow_stor = res[2]
flow_liten = res[0]
flow_stor_conf = res[3]
flow_liten_conf = res[1]


# TODO
# nan -> 0 not accetpable
# Endpoint error function
# Normalize flow?
# visulize flow
# interacitve -> session


import vis


vis.vis_flow(flow_64_48[0, :, :], (48, 64, 3), "64")
vis.vis_flow(flow_stor[0, :, :], (48, 64, 3), "64_pred")

vis.vis_flow(flow_8_6[0, :, :], (6, 8, 3), "8")
vis.vis_flow(flow_liten[0, :, :], (6, 8, 3), "8_pred")


session.close()
'''
