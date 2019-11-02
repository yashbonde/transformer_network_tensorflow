"""
common_layer_fns.py

This module has functions related to tf-layers used in multiple locations

19.09.2019 - @yashbonde
"""
import tensorflow as tf

def shapes_list(inp):
    """
    cleaner handling of tensorflow shapes
    :param inp: input tensor
    :return: list of shapes combining dynamic and static shapes
    """
    shapes_static = inp.get_shape().as_list()
    shapes_dynamic = tf.shape(inp)
    cleaned_shape = [shapes_dynamic[i] if s is None else s for i, s in enumerate(shapes_static)]
    return cleaned_shape

def log_prob_from_logits(logits, reduce_axis = -1):
    return logits - tf.reduce_logsumexp(logits, axis = reduce_axis, keepdims= True)

def sample_with_temprature(logits, temp, sampling_keep_topk = -1):
    if temp == 0.0:
        logits_shape = shapes_list(logits)
        argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis = 1)
        return tf.reshape(argmax, logits_shape[:-1])

    else:
        assert temp > 0.0
        if sampling_keep_topk != -1:
            if sampling_keep_topk <= 0:
                raise ValueError("sampling_keep_topk must be either -1 or +ve")
            vocab_size = shapes_list(logits)[1]

            k_largest = tf.contrib.nn.nth_element(
                logits, n = sampling_keep_topk, reverse = True
            )
            k_largest = tf.tile(tf.reshape(k_largest, [-1, 1]), [1, vocab_size])

            # force every position that is not the in the top k to have probability
            # near 0 by setting the logit to be very negative
            logits = tf.where(
                tf.less_equal(logits, k_largest),
                tf.ones_like(logits) * -1e6,
                logits
            )

        reshaped_logits = (
            tf.reshape(logits, [-1, shapes_list(logits)[-1]]) / temp
        )
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(
            choices, shapes_list(logits)[:logits.get_shape().ndims - 1])

        return choices

def shift_right_3d(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    if pad_value is None:
        shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    else:
        shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
    return shifted_targets

def shift_right_2d(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    if pad_value is None:
        shifted_targets = tf.pad(x, [[0, 0], [1, 0]])[:, :-1]
    else:
        shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1]
    return shifted_targets
