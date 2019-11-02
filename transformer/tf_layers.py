"""
tf_layers.py

All the layers that we use in experiments as functions rather than objects.

14.09.2019 - @yashbonde
"""

import logging
import numpy as np
import tensorflow as tf

from .common_layer_fns import shapes_list

def gelu_activation(inp):
    """
    Gaussian Error Linear Unit (GELU) is a new type of activation function that can
    estimate any of the existing activation values such as Sigmoid, ReLU, ELU, tanh
    while providing superior learning.
    See this [paper](https://arxiv.org/pdf/1606.08415.pdf)

    :param inp: input tensor
    :return:
    """
    out = 1 + tf.tanh(np.sqrt(np.pi) * (inp + 0.044715 * tf.pow(inp, 3)))
    out *= 0.5 * inp
    return out

def softmax_with_reduce_max(inp, axis=-1):
    """
    perform softmax, this is slightly different to the default softmax in tensorflow
    :param inp:
    :param axis:
    :return:
    """
    out = inp - tf.reduce_max(inp, axis=axis, keepdims=True)
    ex = tf.exp(out)
    sm = ex / tf.reduce_sum(ex, axis=axis, keepdims=True)
    return sm


def normalise_tensor(inp, scope, *, axis=-1, epsilon=1e-5):
    """
    Normalize the input values between 0 and 1, then do diagonal affine transform
    :param inp: input tensor
    :param scope: tf variable scope
    :param axis: axis to perform ops on
    :param epsilon: base minimum value
    :return: normalised tensor
    """
    with tf.variable_scope(scope):
        e_dim = inp.get_shape().as_list()[-1]
        g = tf.get_variable('g', [e_dim], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [e_dim], initializer=tf.constant_initializer(0))

        u = tf.reduce_mean(inp, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(inp - u), axis=axis, keepdims=True)
        out = (inp - u) * tf.rsqrt(s + epsilon)
        out = out * g + b

        return out


def split_into_n_states(inp, n):
    """
    reshape last dimension of input tensor from n --> [n, inp.shape[-1]/n]
    :param inp: input tensor
    :param n: number of splits
    :return: reshaped tensor
    """
    *start, m = shapes_list(inp)
    out = tf.reshape(inp, start + [n, m // n])
    return out


def merge_n_states(inp):
    """
    merge the last two dimensions
    :param inp: input tensor
    :return: reshaped tensor
    """
    *start, m, n = shapes_list(inp)
    out = tf.reshape(inp, start + [m * n])
    return out


def conv1d(inp, scope, num_features, weights_init_stddev=0.2):
    """
    1D convolutional block, first reshape input then matmul weights and then reshape

    :param inp: input tensor
    :param scope: tf variable scope
    :param num_features: number of output features
    :param weights_init_stddev: standard deviation value
    :return: processed output
    """
    with tf.variable_scope(scope):
        *start, nx = shapes_list(inp)
        weights = tf.get_variable('w', [1, nx, num_features],
                                  initializer=tf.random_normal_initializer(stddev=weights_init_stddev))
        bias = tf.get_variable('b', [num_features],
                               initializer=tf.constant_initializer(0))

        # reshape input and weights and perform matmul and add bias
        inp_reshaped = tf.reshape(inp, [-1, nx])
        w_reshaped = tf.reshape(weights, [-1, num_features])
        out = tf.matmul(inp_reshaped, w_reshaped) + bias

        out = tf.reshape(out, start + [num_features])
        return out

def ff(inp, scope, num_features, weights_init_stddev=0.2):
    # to fix the name of function from conv1d to ff
    return conv1d(inp, scope, num_features, weights_init_stddev)


def attention_mask(nd, ns, dtype=tf.float32):
    """
    1's in the lower traingle, couting from lower right corner

    This is same as using the tf.matrix_band_part() but it doesn't produce garbage on TPUs

    :param nd:
    :param ns:
    :param dtype:
    :return:
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    out = tf.cast(m, dtype)
    return out


def attention(q, k, v, e_dim, config, mask_future_weights=True, scope='attention'):
    """

    :param q: query tensor
    :param k: key tensor
    :param v: value tensor
    :param e_dim: embedding dimension
    :param config: config object
    :param mask_future_weights: to mask the weights for future
    :param scope: name-scope for function
    :return: output attention tensor
    """
    def split_heads(x):
        out = split_into_n_states(x, config.num_heads)
        out = tf.transpose(out, [0, 2, 1, 3])
        return out

    def merge_heads(x):
        out = merge_n_states(tf.transpose(x, [0, 2, 1, 3]))
        return out

    def mask_attention_weights(w):
        # w should have shape [batches, heads, dst_seq, src_seq], where information flows from scr to dst
        _, _, nd, ns = shapes_list(w)
        b = attention_mask(nd, ns, w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attention(q, k, v, mask=True):
        w = tf.matmul(q, k, transpose_b=True)
        w *= tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        # mask attention weights
        if mask:
            w = mask_attention_weights(w)
        w = softmax_with_reduce_max(w)
        out = tf.matmul(w, v)
        return out

    with tf.variable_scope(scope):
        # projection on convolution + splitting
        q_ = split_heads(conv1d(q, 'q_proj', e_dim))
        k_ = split_heads(conv1d(k, 'k_proj', e_dim))
        v_ = split_heads(conv1d(v, 'v_proj', e_dim))

        # attention
        att = multihead_attention(q_, k_, v_, mask_future_weights)

        # merging
        out = merge_heads(att) + q
        out = conv1d(out, 'conv_projection', e_dim)
        return out


def multilayer_perceptron(inp, scope, hidden_dim):
    """
    MLP

    :param inp: input tensor
    :param scope: tf variable scope
    :param hidden_dim: hidden dimension
    :return: output processed tensor
    """
    with tf.variable_scope(scope):
        nx = inp.shape[-1].value
        out = conv1d(inp, 'convolutional_ff', hidden_dim)
        out = gelu_activation(out)
        out = conv1d(out, 'convolutional_reproj', nx) # re-projection
        return out


def encoder_block(q, scope, config):
    """

    :param q: encoder only has query as input
    :param scope: scope for function
    :param config: config object
    :return: processed tensor
    """
    with tf.variable_scope(scope):
        nx = shapes_list(q)[-1]
        # self attention block
        attn = attention(q, q, q, nx, config, mask_future_weights=False, scope = 'self-attention')
        out = normalise_tensor(attn + q, 'ln_1')

        # mlp
        mlp_out = multilayer_perceptron(out, 'mlp', nx * 4)
        out = normalise_tensor(out + mlp_out, 'ln_2')

        return out


def decoder_block(q, k, v, scope, config):
    """

    :param q: query input
    :param k: key input
    :param v: value input
    :param scope: scope for function
    :param config: config object
    :return: processed tensor
    """
    with tf.variable_scope(scope):
        nx = shapes_list(q)[-1]

        # self attention block
        attn = attention(q, q, q, nx, config, mask_future_weights=True, scope = 'self-attention')
        out = normalise_tensor(attn + q, 'ln_1')

        # external attention block
        attn = attention(out, k, v, nx, config, mask_future_weights=False, scope = 'ext-attention')
        out = normalise_tensor(attn + out, 'ln_2')

        # mlp
        mlp = multilayer_perceptron(out, 'mlp', nx * 4)
        out = normalise_tensor(mlp + out, 'ln_3')

        return out

def past_shape(config, seqlen=None):
    """
    return a list with shape of `past` tensor

    :param config: config object
    :return: list with shape value
    """
    shape = [config.batch_size, config.num_layers, 2, config.num_heads, seqlen,
             config.embedding_dim // config.num_heads]
    return shape


def expand_tile(value, size):
    """
    expand value to size

    :param value: input object to be tiles
    :param size: size to tile the object to
    :return: tiled output
    """
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    out = tf.expand_dims(value, axis=0)
    out = tf.tile(out, [size] + [1, ] * ndims)
    return out


def positions_for(tokens, past_length):
    """
    get positions only for a input tokens

    :param tokens: input tokens
    :param past_length: length of past object
    :return: output
    """
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    out = expand_tile(past_length + tf.range(nsteps), batch_size)
    return out


def embed_sequence(*inp_seq, in_dim, out_dim, scope):
    """perform embedding on input tensors and return tensors and weight matrix"""
    tensors_to_embed = tuple(tensor for tensor in inp_seq)
    with tf.variable_scope(scope):
        w = tf.get_variable(scope + '_matrix', shape = [in_dim, out_dim])
        out = [tf.gather(w, tensor) for tensor in tensors_to_embed]

        return out, w

