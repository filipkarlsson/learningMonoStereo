import tensorflow as tf


def flow_conf_loss_from_flow(pred_flow):

    def optical_flow_conf_loss(true_flow, pred_conf):
        mask = tf.where(tf.is_nan(true_flow), tf.zeros_like(
            true_flow), tf.ones_like(true_flow))

        true_flow_nonans = tf.where(
            tf.is_nan(true_flow), tf.zeros_like(true_flow), true_flow)

        true_conf = tf.math.exp(-tf.abs(pred_flow - true_flow_nonans))

        # axis = -1 ?
        return tf.abs(tf.multiply(pred_conf - true_conf, mask))

    return optical_flow_conf_loss


def flow_loss(true_flow, pred_flow):

    # create mask
    mask = tf.where(tf.is_nan(true_flow), tf.zeros_like(
        true_flow), tf.ones_like(true_flow))

    true_flow_nonan = tf.where(
        tf.is_nan(true_flow), tf.zeros_like(true_flow), true_flow)

    return tf.keras.backend.mean(tf.abs(tf.multiply(mask, pred_flow) - true_flow_nonan), axis=-1)
