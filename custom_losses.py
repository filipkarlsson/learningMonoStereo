import tensorflow as tf
import sys
import os
my = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(my, '../demon/lmbspecialops/python'))
import lmbspecialops


def remove_nan(y_true, y_pred):
    mask_true = tf.where(tf.logical_or(tf.is_nan(y_true), tf.is_inf(y_true)), tf.zeros_like(
            y_true), tf.ones_like(y_true))

    y_true_nonan = tf.where(
            tf.logical_or(tf.is_nan(y_true), tf.is_inf(y_true)), tf.zeros_like(y_true), y_true)

    mask_pred = tf.where(tf.logical_or(tf.is_nan(y_pred), tf.is_inf(y_pred)), tf.zeros_like(
            y_pred), tf.ones_like(y_pred))

    y_pred_nonan = tf.where(tf.logical_or(tf.is_nan(y_pred), tf.is_inf(y_pred)), tf.zeros_like(y_pred), y_pred)

    total_mask = tf.multiply(mask_pred, mask_true)

    y_true_nonan = tf.multiply(y_true_nonan, total_mask)
    y_pred_nonan = tf.multiply(y_pred_nonan, total_mask)

    return y_true_nonan, y_pred_nonan



def flow_conf_loss_from_flow(pred_flow):

    def optical_flow_conf_loss(true_flow, pred_conf):
        mask = tf.where(tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(
            true_flow), tf.ones_like(true_flow))

        true_flow_nonans = tf.where(
            tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(true_flow), true_flow)

        true_conf = tf.exp(-tf.abs(pred_flow - true_flow_nonans))

        # axis = -1 ?
        return tf.abs(tf.multiply(pred_conf - true_conf, mask))

    return optical_flow_conf_loss


def flow_loss(true_flow, pred_flow):
    # create mask
    mask = tf.where(tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(
        true_flow), tf.ones_like(true_flow))

    true_flow_nonan = tf.where(
        tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(true_flow), true_flow)

    return tf.keras.backend.mean(tf.abs(tf.multiply(mask, pred_flow) - true_flow_nonan))


def euclidean_distance_loss(y_true, y_pred):
    y_true, y_pred = remove_nan(y_true, y_pred)
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1) + 1e-10)



def normals_loss_from_depth_gt(depth_true, normals_pred):

    intrinsics = tf.constant([[0.89115971, 1.18821287, 0.5, 0.5]])
    #intrinsics = np.broadcast_to(np.array([[0.89115971, 1.18821287, 0.5, 0.5]]),(batch_size,4))
    depth_true_nchw = tf.transpose(depth_true, [0, 3, 1, 2])

    normals_nchw = lmbspecialops.depth_to_normals(depth_true_nchw, [0.89115971, 1.18821287, 0.5, 0.5], inverse_depth=True)

    # convert to channels last
    normals = tf.transpose(normals_nchw, [0, 2, 3, 1])


    return euclidean_distance_loss(normals, normals_pred)




# A modified version of tf.image.image_gradient
def scale_inv_grad(flow, h):

    if flow.get_shape().ndims != 4:
        raise ValueError('image_gradients expects a 4D tensor '
                         '[batch_size, h, w, d], not %s.', image.get_shape())

    image_shape = tf.shape(flow)
    batch_size, height, width, depth = tf.unstack(image_shape)
    dy = flow[:, h:, :, :] - flow[:, :-h, :, :]
    dy_norm = tf.abs(flow[:, h:, :, :]) + tf.abs(flow[:, :-h, :, :])
    dy_scaled = tf.div(dy, dy_norm + 1e-10)
    #dy_scaled = tf.where(tf.is_nan(dy_scaled), tf.zeros_like(dy_scaled), dy_scaled)

    dx = flow[:, :, h:, :] - flow[:, :, :-h, :]
    dx_norm = tf.abs(flow[:, :, h:, :]) + tf.abs(flow[:, :, :-h, :])
    dx_scaled = tf.div(dx, dx_norm + 1e-10)
    #dx_scaled = tf.where(tf.is_nan(dx_scaled), tf.zeros_like(dx_scaled), dx_scaled)

    # Return tensors with same size as original image by concatenating
    # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
    shape = tf.stack([batch_size, h, width, depth])
    dy_scaled = tf.concat([dy_scaled, tf.zeros(shape, flow.dtype)], 1)
    dy_scaled = tf.reshape(dy_scaled, image_shape)

    shape = tf.stack([batch_size, height, h, depth])
    dx_scaled = tf.concat([dx_scaled, tf.zeros(shape, flow.dtype)], 2)
    dx_scaled = tf.reshape(dx_scaled, image_shape)

    return dx_scaled, dy_scaled


def gradient_loss_8_6(true_flow, pred_flow):
    loss = 0
    for h in [1, 2]:
        grad_x_pred, grad_y_pred = scale_inv_grad(pred_flow, h)
        grad_x_true, grad_y_true = scale_inv_grad(true_flow, h)

        diff_x = tf.square(tf.subtract(grad_x_pred, grad_x_true))
        diff_y = tf.square(tf.subtract(grad_y_pred, grad_y_true))

        element_l2_diff = tf.sqrt(tf.add(diff_x, diff_y) + 1e-10)
        element_l2_diff = tf.where(tf.logical_or(tf.is_nan(element_l2_diff), tf.is_inf(
            element_l2_diff)), tf.zeros_like(element_l2_diff), element_l2_diff)

        #loss = tf.reduce_sum(element_l2_diff)
        loss += tf.keras.backend.mean(element_l2_diff)

    return loss / 2.0


def gradient_loss_64_48(true_flow, pred_flow):
    loss = 0
    for h in [1, 2, 4, 8, 16]:
        grad_x_pred, grad_y_pred = scale_inv_grad(pred_flow, h)
        grad_x_true, grad_y_true = scale_inv_grad(true_flow, h)

        diff_x = tf.square(tf.subtract(grad_x_pred, grad_x_true))
        diff_y = tf.square(tf.subtract(grad_y_pred, grad_y_true))

        element_l2_diff = tf.sqrt(tf.add(diff_x, diff_y) + 1e-10)
        element_l2_diff = tf.where(tf.logical_or(tf.is_nan(element_l2_diff), tf.is_inf(
            element_l2_diff)), tf.zeros_like(element_l2_diff), element_l2_diff)

        #loss = tf.reduce_sum(element_l2_diff)
        loss += tf.keras.backend.mean(element_l2_diff)

    return loss / 5.0


def flow_loss_with_scale_inv_loss_8_6(true_flow, pred_flow):
    mask = tf.where(tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(
        true_flow), tf.ones_like(true_flow))

    true_flow_nonan = tf.where(
        tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(true_flow), true_flow)

    flow_l = flow_loss(true_flow, pred_flow)
    scale_inv_l = gradient_loss_8_6(
        true_flow_nonan, tf.multiply(mask, pred_flow))
    return flow_l + scale_inv_l


def flow_loss_with_scale_inv_loss_64_48(true_flow, pred_flow):
    mask = tf.where(tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(
        true_flow), tf.ones_like(true_flow))

    true_flow_nonan = tf.where(
        tf.logical_or(tf.is_nan(true_flow), tf.is_inf(true_flow)), tf.zeros_like(true_flow), true_flow)

    flow_l = flow_loss(true_flow, pred_flow)
    scale_inv_l = gradient_loss_64_48(
        true_flow_nonan, tf.multiply(mask, pred_flow))
    return flow_l + scale_inv_l


def euclidean_with_gradient_loss(y_true, y_pred):
    y_true, y_pred = remove_nan(y_true, y_pred)

    euclidean_loss = euclidean_distance_loss(y_true, y_pred)
    gradient_loss = gradient_loss_64_48(y_true, y_pred)

    return 1*euclidean_loss + 5*gradient_loss

def mean_abs_error(y_true, y_pred):
    y_true, y_pred = remove_nan(y_true, y_pred)

    return tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)
