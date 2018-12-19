from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, concatenate, Lambda, Conv2DTranspose, Multiply
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import losses
from tensorflow import slice
from tensorflow import image, contrib, zeros, scalar_mul, expand_dims

import numpy as np

import custom_losses
import warp
import helpers


def bootstrap_net(batch_size):

    # 256x192x(3*2)
    image_pair = Input(batch_shape=(batch_size, 192, 256, 6), name='image_pair')

    layer_1a = Conv2D(32, kernel_size=(9, 1), strides=(
        2, 1), activation='relu', padding='same')(image_pair)
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
        x, (0, 0, 0, 0), (-1, -1, -1, 2)), name='optical_flow_output')(layer11)
    confidence_output = Lambda(lambda x: slice(
        x, (0, 0, 0, 2), (-1, -1, -1, 2)), name='confidence_output')(layer11)

    bootstrap_optical_flow = Model(inputs=image_pair, outputs=[
                                   first_pred_optical_flow, first_pred_confidence, optical_flow_output, confidence_output])

    return bootstrap_optical_flow

# input: [second_image, flow]

def bootstrap_net_depth(batch_size):

    # Image pair input branch
    # 256x192x(3*2)
    image_pair = Input(batch_shape=(batch_size, 192, 256, 6), name='image_pair')
    # Input(batch_shape=(batch_size, h, w, c))

    layer_1a = Conv2D(32, kernel_size=(9, 1), strides=(
        2, 1), activation='relu', padding='same', input_shape=(192, 256, 6))(image_pair)
    layer_1b = Conv2D(32, kernel_size=(1, 9), strides=(
        1, 2), activation='relu', padding='same')(layer_1a)

    # 128x96x32
    layer_2a = Conv2D(32, kernel_size=(7, 1), strides=(
        2, 1), activation='relu', padding='same')(layer_1b)
    layer_2b = Conv2D(32, kernel_size=(1, 7), strides=(
        1, 2), activation='relu', padding='same')(layer_2a)

    # Optical flow and conf input branch
    optical_flow_input = Input(batch_shape=(batch_size, 48, 64, 2), name='optical_flow_input')
    optical_flow_conf_input = Input(batch_shape=(batch_size, 48, 64, 2), name='confidense_input')

    second_image = Lambda(lambda x: slice(
        x, (0, 0, 0, 3), (-1, -1, -1, 3)))(image_pair)

    wraped_2nd_image = Lambda(helpers.warped_image_from_flow)(
        [second_image, optical_flow_input])

    # Concatinate wraped 2nd image, optical flow and optical flow conf
    # We dont have depth from flow and motion in bootstrap network

    flow_conf_wraped2nd_merge = concatenate(
        [optical_flow_input, optical_flow_conf_input, wraped_2nd_image])

    layer_wraped_input_a = Conv2D(32, kernel_size=(3, 1), strides=(
        1, 1), activation='relu', padding='same')(flow_conf_wraped2nd_merge)
    # layer3b should be feeded forward later
    layer_wraped_input_b = Conv2D(32, kernel_size=(1, 3), strides=(
        1, 1), activation='relu', padding='same')(layer_wraped_input_a)
    # Concatinate input branches

    concat_input = concatenate([layer_wraped_input_b, layer_2b])

    layer3a = Conv2D(64, kernel_size=(3, 1), strides=(
        1, 1), activation='relu', padding='same')(concat_input)
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

    # End of encoder part
    # ------------------------------

    # Fully connected motion branch begins here
    #
    layer_motion1 = Conv2D(24, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', padding='same')(layer9b)

    flattend_motion_layer = Flatten()(layer_motion1)

    layer_motion2 = Dense(1024)(flattend_motion_layer)

    layer_motion3 = Dense(128)(layer_motion2)

    layer_motion4 = Dense(7)(layer_motion3)

    motion_output = Lambda(lambda x: slice(x, (0, 0), (-1, 6)), name='motion_output')(layer_motion4)

    scale = Lambda(lambda x: slice(x, (0, 6), (-1, -1)))(layer_motion4)

    # End of motion branch
    # ------------------------------

    upconv1 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(
        2, 2),  padding='same', activation='relu')(layer9b)

    upconv1_merge = concatenate([upconv1, layer7b])  # changed

    upconv2 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(
        2, 2),  padding='same', activation='relu')(upconv1)

    upconv2_merge = concatenate([upconv2, layer5b])

    upconv3 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(
        2, 2),  padding='same', activation='relu')(upconv2_merge)

    upconv3_merge = concatenate([upconv3, layer3b])

    layer10 = Conv2D(24, kernel_size=(3, 3), strides=(
        1, 1), activation='relu', padding='same')(upconv3_merge)

    layer11 = Conv2D(4, kernel_size=(3, 3), strides=(
        1, 1), activation=None, padding='same')(layer10)

    # Output :D

    depth = Lambda(lambda x: slice(
        x, (0, 0, 0, 0), (-1, -1, -1, 1)))(layer11)

    scale = Lambda(lambda x: expand_dims(x, 1))(scale)
    scale = Lambda(lambda x: expand_dims(x, 1))(scale)

    #scale = expand_dims(scale, 1)

    depth_output = Multiply(name='depth_output')([depth, scale])

    normals_output = Lambda(lambda x: slice(
        x, (0, 0, 0, 1), (-1, -1, -1, -1)), name='normals_output')(layer11)

    bootstrap_motion_depth_normal = Model(inputs=[image_pair, optical_flow_input, optical_flow_conf_input],
                                          outputs=[motion_output, depth_output, normals_output])

    return bootstrap_motion_depth_normal

    #from tensorflow.python.keras.utils import plot_model
    # plot_model(bootstrap_motion_depth_normal,
    #          to_file='bootstrap_motion_depth_normal.png')


if __name__ == '__main__':
    bootstrap_net_depth(32)
