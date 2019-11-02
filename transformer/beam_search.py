"""
beam_search.py

functions to perform beam search on incoming tensors. Inspired from
    - https://github.com/tensorflow/tensor2tensor

15.09.2019 - @yashbonde
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from .common_layer_fns import shapes_list, log_prob_from_logits

# constants
INF = 1. * 1e7  # default value for infinity
EOS_ID = 1


def get_state_shape_invariant(tensor):
    """Returns the shape of teh tensor but sets middle dims to None"""
    shapes = tensor.shape.as_list()
    shapes[1:len(shapes) - 1] = [None, ] * (len(shapes) - 2)
    return tf.TensorShape(shapes)


def compute_batch_indices(batch_size, beam_size):
    """
    Compute the ith coordinate that contains the batch index for gathers. Batch
    pos is a tensor like [[0,0,0,0], [1,1,1,1], ...]. It says which batch is beam
    item in. This will create the i of i,j coordinate needed for gather
    :param batch_size: bs
    :param beam_size: size of beam
    :return:
    """
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    return tf.reshape(batch_pos, [batch_size, beam_size])


def _merge_beam_dim(tensor):
    """
    Reshapes the first two diemenstions into dingle dimention
    :param tensor: tensor with shape [A, B ...]
    :return: tensor with shape [A*B, ...]
    """
    a, b, *shape = shapes_list(tensor)
    return tf.reshape(tensor, shape=[a * b, *shape])


def _unmerge_beam_dim(tensor, batch_size, beam_dim):
    shape = shapes_list(tensor)
    return tf.reshape(tensor, [batch_size, beam_dim] + shape[1:])


def _expand_to_beam_size(tensor, beam_size):
    """
    Tiles given tensor by beam_size
    :param tensor: tensor to tile
    :param beam_size:  how much to tile
    :return: tiled tensor [bs, beam_size, ...]
    """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size
    return tf.tile(tensor, tile_dims)


def _create_make_unique(inputs):
    """
    Replaces the lower bits of each element with iota. The iota is used to derive
    the index, and also serve the puporse to make each element unique to break ties.

    :param inputs: tensor with rank of 2 ad dtype of tf.float32 [bs, original_size]
    :return: tensor after transformation [bs, original_size]
    """
    if inputs.shape.ndims != 2:
        raise ValueError("Input of `_create_mask_unique` and `top_k_with_unique` "
                         "must be rank 2 but got: {}".format(inputs.shape))

    height, width = inputs.shape
    zeros = tf.zeros([height, width], dtype=tf.int32)

    # count mask is used to mask away the low order bits to ensure that every
    # element is distinct
    log2_ceiling = int(math.ceil(math.log(int(width), 2)))
    next_power_of_two = 1 << log2_ceiling
    count_mask = ~(next_power_of_two - 1)
    count_mask = tf.constant(count_mask)
    count_mask = tf.fill([height, width], count_mask)

    # smallest normal is the bit representation of the smallest positive normal
    # floating point number. The sign is zero, exponent is one, and the fraction is
    # zero
    smallest_normal = 1 << 23
    smallest_normal = tf.constant(smallest_normal, dtype=tf.int32)
    smallest_normal = tf.fill([height, width], smallest_normal)

    # low_bit_mask is used to mask away the sign bit when computing absolute value
    low_bit_mask = ~(1 << 31)
    low_bit_mask = tf.constant(low_bit_mask, dtype=tf.int32)
    low_bit_mask = tf.fill([height, width], low_bit_mask)

    # now we calculate iota
    iota = tf.tile(
        tf.expand_dims(tf.range(width, dtype=tf.int32), 0),
        [height, 1]
    )

    # compare the absolute value with positive zero and handle negative zero
    inputs = tf.bitcast(inputs, tf.int32)
    abs_ = tf.bitwise.bitwise_and(inputs, low_bit_mask)
    if_zero = tf.equal(abs_, zeros)
    smallest_normal_preserving_sign = tf.bitwise.bitwise_or(
        inputs,
        smallest_normal
    )
    input_no_zeros = tf.where(
        if_zero,
        smallest_normal_preserving_sign,
        inputs
    )

    # discard al lower bits and replace with iota
    and_ = tf.bitwise.bitwise_and(input_no_zeros, count_mask)
    or_ = tf.bitwise.bitwise_or(and_, iota)
    out = tf.bitcast(or_, tf.float32)

    return out


def _create_top_k_unique(inputs, k):
    """
    creates the top k values in sorted order with indices

    :param inputs: a tensor with rank of 2. [bs, original_size]
    :param k: integer for the largest element
    :return:
    """

    height, width = inputs.shape

    # make the negative infinity mask and clean inputs
    neg_inf = tf.constant(-np.inf, tf.float32)
    neg_inf = tf.fill([height, width], neg_inf)
    inputs = tf.where(tf.is_nan(inputs), neg_inf, inputs)

    # selected the current largest value k times and keep them in topk. The
    # selected largest values are marked as the smallest to avoid being used again
    tmp = inputs
    topk = tf.zeros([height, k], dtype=tf.float32)
    for i in range(k):
        kth_order_stats = tf.reduce_max(tmp, axis=1, keepdims=True)
        k_mask = tf.tile(tf.expand_dims(
            tf.equal(tf.range(k), tf.fill([k], i)),
            0
        ), [height, 1])
        topk = tf.where(k_mask, tf.tile(kth_order_stats, [1, k]), topk)
        ge = tf.greater_equal(inputs, tf.tile(kth_order_stats, [1, width]))
        tmp = tf.where(ge, neg_inf, inputs)

    # continuing
    log2_ceiling = int(math.ceil(math.log(float(int(width)), 2)))
    next_power_of_two = 1 << log2_ceiling
    count_mask = next_power_of_two - 1
    mask = tf.constant(count_mask)
    mask = tf.fill([height, k], mask)
    topk_idx = tf.bitcast(topk, tf.int32)
    topk_idx = tf.bitwise.bitwise_and(topk_idx, mask)

    return topk, topk_idx


def top_k_with_unique(inputs, k):
    """
    Finds the values and indices of the k-largest entries. Instead of doing sort
    like `tf.nn.top_k`, this function finds the max value k times. The running time
    is proportional to k, which is to be faster when k is small. The current
    implementation supports onle inputs of rank 2. In addition, iota is used to
    replace the lower bits of each element, this makes the selection more stable
    when there are equal elements. The overhead is that values are approximated.

    :param inputs: input tensor with rank-2 [bs, original_size]
    :param k: integer for top k elements

    :return top_values: a tensor with the k largest elements in sorted order [bs, k]
    :return indices: a tensor, indices of the top k elements [bs, k]
    """
    unique_values = _create_make_unique(tf.cast(inputs, tf.float32))
    top_values, indices = _create_top_k_unique(unique_values, k)
    top_values = tf.cast(top_values, inputs.dtype)
    return top_values, indices


def compute_topk_scores_and_seq(sequences,
                                scores,
                                scores_to_gather,
                                flags,
                                beam_size,
                                batch_size,
                                prefix="default",
                                states_to_gather=None):
    # first we gather the top k indices
    _, top_k_idx = top_k_with_unique(scores, k=beam_size)

    # the next steps are to create coordinates for tf.gather_nd to pull out the
    # top k sequences from sequence based on scores. batch_pos is a tensor like
    # [[0,0,0,0], [1,1,1,,1], ...]. It says which batch is beam items is in. This
    # will create the `i` for `i,j` coordinate, needed for gather
    batch_pos = compute_batch_indices(batch_size, beam_size)

    # top_coordinates will give us the actual coordinates to gather. Stacking will
    # create a tensor fo dimension batch * beam * 2, where the last dimension
    # contains the `i,j` gathering coordinates.
    top_coordinates = tf.stack([batch_pos, top_k_idx], axis=2)

    # do gathers
    topk_seq = tf.gather_nd(sequences, top_coordinates, name=prefix + '_topk_seq')
    topk_flags = tf.gather_nd(flags, top_coordinates, name=prefix + '_topk_flags')
    topk_scores = tf.gather_nd(scores_to_gather, top_coordinates,
                               name=prefix + '_topk_scores')

    if states_to_gather:
        topk_states = nest.map_structure(
            lambda state: tf.gather_nd(state, top_coordinates,
                                       name=prefix + '_topk_states'),
            states_to_gather
        )
    else:
        topk_states = states_to_gather

    return topk_seq, topk_scores, topk_flags, topk_states


def beam_search(config, model_namespace,
                symbols_to_logits_fn,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                initial_id = None,
                initial_ids = None,
                eos_id=EOS_ID,
                states=None,
                stop_early=True):
    """
    Beam Search with length penalties. Requires a function that can take the
    currently decoded symbols and return the logits for the next symbol. This
    implementation is inspired by https://arxiv.org/abs/1609.08144.

    When running, the beam search steps and be visualised using `tfdbg` to watch
    the operations generating the output ids for each beam step. These ops have
    the pattern:
        (alive|finished)_topk_(seq, scores)

    Ops marked `alive` represent the new beam sequences that will be processed
    in the next step. Operations marked `finished` represent the completed beam
    sequences, which may be padded if no beams finished.

    Ops marked `seq` store the full beam sequence for the time step. Ops marked
    `score` store the sequence's final log scores.

    THe beam search steps will be processed sequentially in order, so when capturing
    opbserved from these ops, tensors, clients can make assumptions about which
    step is being recorded.

    WARNING: Assumes 2nd dimension of tensors in `states` and not invariant, this
    means that the shape of second dimension of these tensors will not be available
    (set to None) inside symbols_to_logits_fn.

    :param symbols_to_logits_fn: Interface to the model, to provide logits.
        Should take [batch_size, decoded_ids] and return [batch_size, vocab_size]
    :param initial_ids: Ids to start off the decoding, this will be the first thing
        handed to symbols_to_logits_fn (after expanding to beam size)
        [batch_size]
    :param beam_size: Size of the beam.
    :param decode_length: Number of steps to decode for.
    :param vocab_size: Size of the vocab, must equal the size of the logits
        returned by symbols_to_logits_fn
    :param alpha: alpha for length penalty.
    :param eos_id: ID for end of sentence.
    :param states: dict (possibly nested) of decoding states.
    :param stop_early: a boolean - stop once best sequence is provably determined.

    :returns: Tuple of (decoded beams [batch_size, beam_size, decode_length]
        decoding probabilities [batch_size, beam_size])
    """

    if initial_id is None:
        assert initial_ids is not None, "need to provide either `initial_ids` or `initial_id`"
    elif initial_ids is None:
        assert initial_id is not None, "need to provide either `initial_ids` or `initial_id`"
    else:
        raise ValueError("need to provide either `initial_ids` or `initial_id`")

    batch_size = shapes_list(initial_ids)[0]

    # assume initial IDs log prob are 1. and expand to beam_size
    init_log_probs = tf.constant([[0.] + [-INF] * (beam_size - 1)])
    alive_log_probs = tf.tile(init_log_probs, [batch_size, 1])

    # expand each batch and state to beam_size
    alive_seq = _expand_to_beam_size(initial_ids, beam_size) # [bs, beam_size]
    alive_seq = tf.expand_dims(alive_seq, axis=2)  # [bs, beam_size, 1]
    if states:
        states = nest.map_structure(
            lambda state: _expand_to_beam_size(state, beam_size),
            states
        )
    else:
        states = {}

    # finished will keep track of al the sequences that have finished so far
    # finished log probs will be negative inf in the beginning
    finished_seq = tf.zeros(shapes_list(alive_seq), tf.int32)
    # setting scores of the initial to negatie inf
    finished_scores = tf.ones([batch_size, beam_size]) * -INF
    # finished_flags will be keep track of booleans
    finished_flags = tf.zeros([batch_size, beam_size], tf.bool)

    def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                      curr_scores, curr_finished):
        """
        Given sequences and scores, will gather the top_k = beam size sequences

        :param finished_seq: Current finished sequences.
            [batch_size, beam_size, current_decoded_length]
        :param finished_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param finished_flags: finished bools for each of these sequences.
           [batch_size, beam_size]
        :param curr_seq: current topk sequence that has been grown by one position.
            [batch_size, beam_size, current_decoded_length]
        :param curr_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param curr_finished: Finished flags for each of these sequences.
            [batch_size, beam_size]

        :returns: Tuple of (Topk sequences based on scores,
                            log probs of these sequences,
                            Finished flags o}f these sequences)
        """
        # print('******* Inside grow finished')
        _z = tf.zeros([batch_size, beam_size, 1], dtype=tf.int32)
        finished_seq = tf.concat(
            [finished_seq, _z],
            axis=2
        )

        # set the scores of the unfinished seq in curr_seq to large negative values
        curr_scores += (1. - tf.cast(curr_finished, tf.float32)) * -INF

        # concatenating the sequences and scores along the beam axis
        curr_finished_seq = tf.concat([finished_seq, curr_seq], axis=1)
        curr_finished_scores = tf.concat([finished_scores, curr_scores], axis=1)
        curr_finished_flags = tf.concat([finished_flags, curr_finished], axis=1)

        return compute_topk_scores_and_seq(
            sequences=curr_finished_seq,
            scores=curr_finished_scores,
            scores_to_gather=curr_finished_scores,
            flags=curr_finished_flags,
            beam_size=beam_size,
            batch_size=batch_size,
            prefix="grow_finished"
        )

    def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished, states):
        """
        Given sequences and scores, will gather the top_k = beam_size sequences

        :param curr_seq: current topk sequence that has been grown by one position.
            [batch_size, beam_size, i+1]
        :param curr_scores: scores for each of these sequences.
            [batch_size, beam_size]
        :param curr_log_probs: log probs for each of these sequences.
            [batch_size, beam_size]
        :param curr_finished: Finished flags for each of these sequences.
            [batch_size, beam_size]
        :param states: dict (possibly nested) of decoding states.

        :returns: Tuple of (Topk sequences based on scores,
                            log probs of these sequences,
                            Finished flags of these sequences)
        """
        # print('******* Inside grow alive')
        curr_scores += tf.cast(curr_finished, tf.float32) * -INF
        return compute_topk_scores_and_seq(
            sequences=curr_seq,
            scores=curr_scores,
            scores_to_gather=curr_log_probs,
            flags=curr_finished,
            beam_size=beam_size,
            batch_size=batch_size,
            prefix="grow_alive",
            states_to_gather=states
        )

    def grow_topk(i, alive_seq, alive_log_probs, states):
        """
        Inner beam search loop.

        This functions takes the current alive sequences and grows them to topk
        sequences where k = 2 * beam. We use 2 * beam because we could have
        beam_size number of sequences that might hit <EOS> and there will be no
        alive sequences to continue. With 2 * beam_size this will not happen.
        This relies on assumption that the vocab_size > beam_size. If this is true,
        we'll have at least beam_size non <EOS> extensions if we extract the next
        top 2 * beam words. Length penalty is given by
            = (5 + len(decode) / 6) ^ -alpha

        :param i: loop index
        :param alive_seq: topk sequences decoded so far [bs, beam_size, i + 1]
        :param alive_log_probs: probabilities of those sequences
            [bs, beam_size]
        :param states: dict (possibly nested) of decoding states.

        :returns: Tuple of
            (Topk sequences extended by the next word,
             The log probs of these sequences,
             The scores with length penalty of these sequences,
             Flags indicating which of these sequences have finished decoding,
             dict of transformed decoding states)
        """
        # print('******* Inside grow topk')
        flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])

        if states:
            flat_states = nest.map_structure(_merge_beam_dim, states)
            flat_logits, flat_states = symbols_to_logits_fn(flat_ids, i, flat_states)
            states = nest.map_structure(
                lambda t: _unmerge_beam_dim(t, batch_size, beam_size),
                flat_states
            )
        else:
            flat_logits = symbols_to_logits_fn(model_namespace, config, flat_ids)

        logits = tf.reshape(flat_logits, [batch_size, beam_size, -1])

        # convert logits to normalised log probs
        cadidate_log_probs = log_prob_from_logits(logits)

        # multiply the probabilities by current probabilities of the beam.
        # (bs, beam_size, vocab_size) + (bs, beam_size, 1)
        log_probs = cadidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

        length_penalty = tf.pow(((5. + tf.cast(i + 1, tf.float32)) / 6.), alpha)
        curr_scores = log_probs / length_penalty

        # flatten out the probabilities to a list of possibilities
        flat_curs_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])

        # get new scores
        topk_scores, topk_ids = top_k_with_unique(flat_curs_scores, k=beam_size * 2)
        topk_log_probs = topk_scores * length_penalty

        topk_beam_index = topk_ids // vocab_size
        topk_ids %= vocab_size  # unflatten the ids

        # next three steps are to create coordinates for tf.gather_nd to pull
        # out the correct sequences from id's that we need to grow. We will also
        # use the coordinates to gather the booleans of the beam items that survived
        batch_pos = compute_batch_indices(batch_size, beam_size * 2)

        # top beams will give us the actual coordiantes to gather, stacking will
        # create tensor of dimension bs * beam_size * 2, where the last dimension
        # containg s the i,j gathering coordiantes
        topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2)

        # gather up the most probable 2 * beams both for ids and
        # finished_in_alive bools
        topk_seq = tf.gather_nd(alive_seq, topk_coordinates)

        if states:
            states = nest.map_structure(
                lambda state: tf.gather_nd(state, topk_coordinates),
                states
            )

        # print(states)

        # append the most probable alive
        topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)],
                             axis=2)

        topk_finished = tf.equal(topk_ids, eos_id)

        return topk_seq, topk_log_probs, topk_scores, topk_finished, states

    def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                   finished_flags, states):
        """
        Inner beam search loop. There are three groups of tensors, alive,
        finished and topk. The alive group contains information about the current
        alive sequences. The topk group contains information about alive + topk
        current decoded words and the finished sequences contains information about
        finished sequences, i.e. the ones that are decoded to <EOS>. These are
        what we return.

        The general beam search algorithm is as follows:
        While we haven't terminated:
            1. Grow the current alive sequence to get beam * 2 topk sequences
            2. Among the topk, keep the top beam_size ones that haven't reached
                <EOS> into alive
            3. Among the topk, keep the top beam_size onces that have reached
                <EOS> into finished
        Repeat

        To make things simple we using fixed size tensors, we will end up inserting
        unfinished sequences into finished at the beginning. To stop that we add
        -ve INF to score of the unfinished sequence so that when a true unfinished
        sequence does appear, it will have higher score that all the unfinished ones.

        :param i: loop index
        :param alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
        :param alive_log_probs: probabilities of the beams [batch_size, beam_size]
        :param finished_seq: Current finished sequences [batch_size, beam_size, i+1]
        :param finished_scores: scores for each of these sequences
            [batch_size, beam_size]
        :param finished_flags: finished bools for each of these sequences.
            [batch_size, beam_size]
        :param states: dict (possibly nested) of decoding states.

        :returns: Tuple of
            (Incremented loop index
             New alive sequences,
             Log probs of the alive sequences,
             New finished sequences,
             Scores of the new finished sequences,
             Flags indicating which sequence in finished as reached EOS,
             dict of final decoding states)
        """
        # print('******* Inside inner_loop')

        # step 1: grow the curent sequence
        topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
            i, alive_seq, alive_log_probs, states
        )

        # print(states)
        # step 2: among the top-k, keep the top beam_size alive ones
        alive_seq, alive_log_probs, _, states = grow_alive(
            topk_seq, topk_scores, topk_log_probs, topk_finished, states
        )

        # print(states)
        # step 3: among the top-k, keep the to beam_size finished ones
        finished_seq, finished_scores, finished_flags, _ = grow_finished(
            finished_seq, finished_scores, finished_flags, topk_seq,
            topk_scores, topk_finished
        )

        return (
            i + 1,
            alive_seq,
            alive_log_probs,
            finished_seq,
            finished_scores,
            finished_flags,
            states
        )

    def _is_not_finished(i, unused_alive_seq, alive_log_probs,
                         unused_finished_seq, finished_scores,
                         unused_finished_in_finished, unused_states):
        """
        checking terminating condition. We terminate when we decoded up to
        decode_length or the lowest scoring item in finished has a greater score
        that the highest prob item in alive divided by max length penalty.
        NOTE: There are multiple inputs, which are not used but required for
        using it properly in tf.while_loop()

        :param i: loop index
        :param alive_log_probs: probabilities of the beams. [batch_size, beam_size]
        :param finished_scores: scores for each of these sequences.
            [batch_size, beam_size]

        :returns: bool
        """
        max_length_penalty = tf.pow(
            (5. + tf.cast(decode_length, tf.float32)) / 6.,
            alpha
        )

        # the best possible score of the most likely alive sequence
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

        if not stop_early:
            """by considering the min score (in the top N beams) we ensure that
            the decoder will keep decoding until there is at least one beam
            (in the top N) that can be improved (w.r.t. the alive beams).
            any unfinished beam will have score -ve INF - thus the min will always
            be -be INF if there is atleast one unifinished beam -
            which means the bound_is_met condition cannot be true in this case"""
            lowest_score_of_finished_in_finished = tf.reduce_min(finished_scores)
        else:
            """by taking the max score we only care about the first beam; as soon
            as thif first beam cannot be beaten from alive beams the beam decoder
            can stop. Similarly to the above, if the top beam is not complete
            its finished_score is iv INF, thus it will not activate the bound_is_met
            condition, i.e. the decoder will keep on going. Note we need to find
            the max for every sequence separately - so, we need to keep the
            batch dimension"""
            lowest_score_of_finished_in_finished = tf.reduce_max(finished_scores,
                                                                 axis=1)

        bound_is_met = tf.reduce_all(
            tf.greater(lowest_score_of_finished_in_finished,
                       lower_bound_alive_scores)
        )

        # true if i < decode_length and not bound_is_met
        out_cond = tf.logical_and(
            tf.less(i, decode_length), tf.logical_not(bound_is_met)
        )

        return out_cond

    # outside the functions
    inner_shape = tf.TensorShape([None, None, None])
    state_struc = nest.map_structure(get_state_shape_invariant, states)

    # using the while loop
    _, alive_seq, alive_log_probs, finished_seq, finished_scores, \
    finished_flags, states = tf.while_loop(
        _is_not_finished,
        inner_loop,
        (
            tf.constant(0), alive_seq, alive_log_probs, finished_seq,
            finished_scores, finished_flags, states
        ),
        shape_invariants=(
            tf.TensorShape([]),
            inner_shape,
            alive_log_probs.get_shape(),
            inner_shape,
            finished_scores.get_shape(),
            finished_flags.get_shape(),
            state_struc
        ),
        parallel_iterations=1,
        back_prop=False
    )

    # once outside the loop reshape the outputs
    alive_seq.set_shape((None, beam_size, None))
    finished_seq.set_shape((None, beam_size, None))

    # Accounting for corner case: It's possible that no sequence in alive for a
    # particular batch item ever reached EOS. In that case, we should just copy the
    # contents of alive for that batch item.
    # if tf.reduce_any(finished_flags, 1) == 0, means that not sequence for that
    # batchindex has reached EOS. We need to do that same for the scores as well
    finished_seq = tf.where(
        tf.reduce_any(finished_flags, 1),
        finished_seq,
        alive_seq
    )
    finished_scores = tf.where(
        tf.reduce_any(finished_flags, 1),
        finished_scores,
        alive_log_probs
    )

    return finished_seq, finished_scores, states
