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
        g = tf.get_variable(
            'g', [e_dim], initializer=tf.constant_initializer(1))
        b = tf.get_variable(
            'b', [e_dim], initializer=tf.constant_initializer(0))

        u = tf.reduce_mean(inp, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(inp - u), axis=axis, keepdims=True)
        out = (inp - u) * tf.math.rsqrt(s + epsilon)
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


def ff(inp, scope, num_features, weights_init_stddev=0.2, return_param=False):
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
        if not return_param:
            return out
        return out, weights, bias


def attention_mask_future(nd, ns, dtype=tf.float32):
    """
    1's in the lower traingle, couting from lower right corner
    This is same as using the tf.matrix_band_part() but it doesn't produce garbage on TPUs
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    out = tf.cast(m, dtype)
    return out


def attention_mask_key(key_mask, num_tile, dtype=tf.float32):
    """
    We do not want our outputs to be affeted by the garbage in <PAD> so we take the
    key mask and get it to 
    """
    key_mask = tf.cast(key_mask, dtype)
    key_mask = tf.tile(key_mask, [num_tile, 1])  # (h*N, seqlen)
    key_mask = tf.expand_dims(key_mask, 1)  # (h*N, 1, seqlen)
    mask = key_mask * - tf.cast(1e10, dtype)
    return mask


def attention(q, k, v, e_dim, config, key_mask, mask_future_weights=True, scope='attention', training=False):
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
        dim0, dim1, *rest_shapes = shapes_list(out)
        return tf.reshape(out, [dim0 * dim1, *rest_shapes])

    def merge_heads(x):
        dim0, *rest_shapes = shapes_list(x)
        x_reshaped = tf.reshape(
            x, [dim0 // config.num_heads, config.num_heads, *rest_shapes])
        out = merge_n_states(tf.transpose(x_reshaped, [0, 2, 1, 3]))
        return out

    def mask_attention_weights_future(w):
        # w should have shape [batches * heads, dst_seq, src_seq], where information flows from scr to dst
        _, nd, ns = shapes_list(w)
        b = attention_mask_future(nd, ns, w.dtype)
        b = tf.reshape(b, [1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def mask_attention_weights_key(w, key_mask):
        '''What we are trying here is to apply the boolean mask that comes with t padding so we do not
        want it to ocus on that. Shapes are: 
            w [qk_t]: [batch_size, num_heads, emb_dim, emb_dim]
            key_mask: [batch_size, seqlen]
        '''
        dim0, nd, ns = shapes_list(w)
        # V true because `w` may change during beam search
        num_batches = shapes_list(key_mask)[0]
        b = attention_mask_key(key_mask, dim0 // num_batches, w.dtype)
        return w + b

    def multihead_attention(q, k, v, key_mask, future_mask=True):
        '''q -----> [N*h, seqlen, edim/h]
        key_mask -> [N, seqlen]'''
        w = tf.matmul(q, k, transpose_b=True)
        w *= tf.math.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        # mask attention weights
        w = mask_attention_weights_key(w, key_mask)
        # tf.summary.image('attention_mask', tf.expand_dims(tf.expand_dims(w[0], 0), -1))
        if future_mask:
            w = mask_attention_weights_future(w)
            # tf.summary.image('future_mask', tf.expand_dims(tf.expand_dims(w[0], 0), -1))
        w = tf.nn.softmax(w)
        out = tf.matmul(w, v)
        return out

    with tf.variable_scope(scope):
        # projection on convolution + splitting
        q_ = split_heads(ff(q, 'q_proj', e_dim))  # [N*n, seqlen, edim/h]
        k_ = split_heads(ff(k, 'k_proj', e_dim))  # [N*n, seqlen, edim/h]
        v_ = split_heads(ff(v, 'v_proj', e_dim))  # [N*n, seqlen, edim/h]

        # attention
        out = merge_heads(multihead_attention(
            q_, k_, v_, key_mask, mask_future_weights))
        return tf.layers.dropout(out, 0.3, training=training)


def multilayer_perceptron(inp, scope, hidden_dim, act='relu', training=False):
    """
    MLP
    :param inp: input tensor
    :param scope: tf variable scope
    :param hidden_dim: hidden dimension
    :return: output processed tensor
    """
    ACT = {
        "relu": tf.nn.relu,
        "gelu": gelu_activation
    }
    with tf.variable_scope(scope):
        nx = inp.shape[-1].value
        out = ff(inp, 'convolutional_ff', hidden_dim)
        if act in ACT:
            out = ACT[act](out)
        out = ff(out, 'convolutional_reproj', nx)  # re-projection
        return tf.layers.dropout(out, 0.3, training=training)

        def encoder_block(q, ext_mask, scope, config):
    """

    :param q: encoder only has query as input
    :param ext_mask: external masking --> pad masking
    :param scope: scope for function
    :param config: config object
    :return: processed tensor
    """
    with tf.variable_scope(scope):
        nx = shapes_list(q)[-1]
        # self attention block
        attn = attention(q, q, q, nx, config, ext_mask,
                         mask_future_weights=False, scope='self-attention')
        out = normalise_tensor(attn + q, 'ln_1')

        # mlp
        mlp_out = multilayer_perceptron(out, 'mlp', nx * 4)
        out = normalise_tensor(out + mlp_out, 'ln_2')

        return out


def decoder_block(q, k, v, enc_mask, dec_mask, scope, config):
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
        attn = attention(q, q, q, nx, config, dec_mask,
                         mask_future_weights=True, scope='self-attention')
        out = normalise_tensor(attn + q, 'ln_1')

        # external attention block
        attn = attention(out, k, v, nx, config, enc_mask,
                         mask_future_weights=False, scope='ext-attention')
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
        w = tf.get_variable(scope + '_matrix', shape=[in_dim, out_dim])
        out = [tf.gather(w, tensor) for tensor in tensors_to_embed]
        return out, w


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def label_smoothing(inputs, epsilon=0.1):
    return ((1-epsilon) * inputs) + (epsilon / shapes_list(inputs)[-1])


def get_sine_cosine_embedding_matrix(maxlen, edim):
    position_enc = np.array([
        [pos / np.power(10000, (i-i % 2)/edim) for i in range(edim)]
        for pos in range(maxlen)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)
