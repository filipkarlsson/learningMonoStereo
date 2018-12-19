import os
import sys
my = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(my, '../demon/lmbspecialops/python'))
import lmbspecialops
import tensorflow as tf
import warp

def flow_from_depth(inputs):
    depth = inputs[0]
    motion = inputs[1]
    intrinsics = [0.89115971, 1.18821287, 0.5, 0.5]
    depth_nchw = tf.transpose(depth, [0, 3, 1, 2])
    rotation = tf.slice(motion, [0,0], [-1, 3])
    translation = tf.slice(motion, [0,3], [-1, 3])
    flow_ncwh = lmbspecialops.depth_to_flow(depth_nchw, intrinsics, rotation, translation, rotation_format='angleaxis3', inverse_depth=True, normalize_flow=True)

    return tf.transpose(flow_ncwh, [0, 2, 3, 1])


def warped_image_from_flow(inputs):
    second_image = inputs[0]
    flow = inputs[1]
    second_image_48_64 = tf.image.resize_images(second_image, (48, 64))

    print(second_image_48_64)
    print(flow)

    # Create warped 2nd image
    # This layer requires batch size to be specified for some reason
    return warp.dense_image_warp(
        second_image_48_64, flow)
