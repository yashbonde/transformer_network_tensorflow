"""main trainer file for all of our models. This has the tensorboard support as well
as the checkpoint system. I still haven't figured out an easy way to use the embedding
visualisation in a easy fashion where our tokens keep changing.
06.11.2019 - @yashbonde
"""
import sys
# sys.stdout.close() # this is used when there are things blocked in the network

# import dependencies
from transformer.ops_util import ModelConfig
from transformer.data_load import get_batch, Encoder
from transformer.tf_layers import positions_for
from transformer.beam_search import beam_search

# import tensorflow.python.debug as tf_debug
from tqdm import tqdm
import sentencepiece as spm
import tensorflow as tf
import numpy as np
import os
import json
import logging

# custom config
config = ModelConfig(
    description='Master script to train transformer models. Using noam learning rate scheme.')
# future cases to handle --> model architecture to use, currently it is the baseline model
config.add_arg('--mode', type=str, default='v1', required=True)
config.add_arg('--filepath', type=str,
               help='path to file with samples', required=True)
config.add_arg('--spm_model', type=str,
               help='path to SPM model', required=True)

# model config
config.add_arg('--arch', type=str, default='baseline', help='model architecture. For refrence goto README.md')
config.add_arg('--num_layers', type=int, default=4, help='this is the number of stacks to use')
config.add_arg('--num_heads', type=int, default=1, help='number of heads in MHA module')
config.add_arg('--embedding_dim', type=int, default=30, help='embedding dimension value')
config.add_arg('--cntx_len', type=int, default=60, help='maximum length to use for encoder')
config.add_arg('--use_inverse_embedding', type=bool, default=False, help='to use context embedding matrix for word prediction also')
config.add_arg('--save_folder', type=str, default='./bubble', help='path to save the model')
config.add_arg('--opt', type=str, default='adam', help='Optimizer to use for training the model [adam, sgd, rmsprop]')
config.add_arg('--lr', type=float, default=0.05, help='learning rate to use for training')
config.add_arg('--batch_size', type=int, default=128, help='batch size for training the model')
config.add_arg('--beam_size', type=int, default=5, help='beam size used for decoding, this is the number of options we get during decoding')
config.add_arg('--length_penalty_factor', type=int, default=1.0, help='we want to have a length penalty after which the model should stop decoding')
config.add_arg('--max_decode_length', type=int, default=60, help='maximum decoder tensor lenght')

# train ops
config.add_arg('--num_epochs', type=int, default=100, help='number of epochs to train for')
config.add_arg('--log_every', type=int, default=5, help='save in a dump txt file every this steps')
config.add_arg('--max_saves', type=int, default=2, help='latest number of models to save')
config.add_arg('--seed', type=int, default=0, help='random number seed value') config.parse_args()

# now we add some other values that will be grouped with the
config.add_key_value('training', True)
config.add_key_value('max_encoder_length', config.cntx_len)
with open(config.spm_model.replace('.model', '.vocab'), 'r', encoding='utf-8') as f:
    config.add_key_value('vocab_size', len(f.readlines()))
config.path = os.path.join(config.save_folder, 'model_config.json')
if not os.path.exists(config.save_folder):
    os.makedirs(config.save_folder)
config.save_json()

if config.mode == 'v1':
    print('\n\n-------- Using V1 --------\n\n')
    from transcend import baseline as transformer_model
elif config.mode == 'v2':
    print('\n\n-------- Using V2 --------\n\n')
    from transcend import baselinev2 as transformer_model

if config.seed > 0:
    # feed seed values
    tf.random.set_random_seed(config.seed)
    np.random.seed(config.seed)

# data loader
batches, num_batches, num_samples = get_batch(
    fpath=config.filepath,
    encoder_fpath=config.spm_model,
    maxlen1=config.max_encoder_length,
    maxlen2=config.max_decode_length,
    batch_size=config.batch_size,
    shuffle=True)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(
    batches.output_types, batches.output_shapes)
xs, ys = iter.get_next()
train_init_op = iter.make_initializer(batches)

# model
spm_encoder = Encoder(config.spm_model)
config.add_key_value('pad_id', spm_encoder.pad_id)
model_namespace = transformer_model.transformer(
    config=config,
    encoder_placeholder=xs[0],
    target_placeholder=ys[0],
    training=True)

# beam search decoder
initial_ids = tf.constant([spm_encoder.bos_id,] * config.batch_size, dtype=tf.int32)


def symbols_to_logits_fn(model, config, decoder_tensor, debug=False):
    '''We basically need to run the complete decoder function
    :param model: namespace returned from function
    :param decoder_tensor: [batch_size * beam_size, decoded_length]
    :return new_ids: [batch_size * beam_size, vocab_size]
    '''
    print('^^^^^^ decoder_tensor: {}'.format(decoder_tensor))
    decoder_gather = tf.gather(
        model.context_embedding, decoder_tensor
    ) * (config.embedding_dim ** 0.5)
    decoder_gather += tf.gather(model.position_embedding,
                                positions_for(decoder_tensor, past_length=0))
    print('>>>>> {}'.format(decoder_gather))
    encoder_tiled = tf.tile(model.encoder_embedding, [config.beam_size, 1, 1])
    print('>>> encoder_tiled: {}'.format(encoder_tiled))
    local_decoder_pad_mask = tf.math.equal(
        decoder_tensor, config.pad_id, name='beam_decoder_pad_mask')
    print('>>>> local_decoder_pad_mask: {}'.format(local_decoder_pad_mask))
    decoder_out_func = transformer_model.decoder_fn(config=config,
                                                    dec_out=decoder_gather,
                                                    enc_out=encoder_tiled,
                                                    encoder_pad_mask=model.encoder_pad_mask,
                                                    decoder_pad_mask=local_decoder_pad_mask)  # [bs, None, embedding_dim]
    print('>>>> decoder_out_func: {}'.format(decoder_out_func))
    # [bs, None, vocab_size]
    decoder_out = tf.matmul(decoder_out_func, model.fproj_w, transpose_b=True)
    print('>>>> decoder_out: {}'.format(decoder_out))
    decoder_out_last_step = decoder_out[:, -1, :]  # [bs, vocab_size]
    print('>>> decoder_out_last_step: {}'.format(decoder_out_last_step))
    return decoder_out_last_step


finished_seq, finished_scores, _ = beam_search(
    config=config,
    model_namespace=model_namespace,
    symbols_to_logits_fn=symbols_to_logits_fn,
    initial_ids=initial_ids,
    initial_id=None,
    beam_size=config.beam_size,
    decode_length=config.max_decode_length,
    vocab_size=config.vocab_size,
    alpha=config.length_penalty_factor,
    eos_id=spm_encoder.eos_id,
    states=None,
    stop_early=True)

print('** finished_scores: {}'.format(finished_scores))
print('** finished_seq: {}'.format(finished_seq))
print('========== DONE ===========')
# exit()

# training
with tf.Session() as sess:
    # tensorboard + saver support
    train_writer = tf.summary.FileWriter(config.save_folder, sess.graph)
    merged_summary = tf.summary.merge_all()
    model_save_path = config.save_folder + '/{}'.format(config.arch)
    saver_ = tf.train.Saver(max_to_keep=config.max_saves)
    global_step = 0
    # initialisers
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    print('\n----- STARTING TRAINING -----')

    # other
    dump_file = os.path.join(config.save_folder, 'samples.txt')
    ops = [model_namespace.loss, model_namespace.train_step,
           merged_summary, model_namespace.encoder_embedding]
    for eidx in range(config.num_epochs):
        print('---- EPOCH {} ----'.format(eidx))
        # ep_loss = []
        for bidx in tqdm(range(num_batches)):
            _loss, _, _summary, enc_emb = sess.run(ops)
            train_writer.add_summary(_summary, global_step)
            global_step += 1
            # ep_loss.append(_loss)
            # print('---> Loss: {}'.format(_loss))

        # save at the end of every epoch
        if eidx % config.log_every == 0 or eidx == config.num_epochs - 1:  # save in a dump file
            # epm = sess.run(model_namespace.decoder_pad_mask)
            # print(epm.shape)

            print('Writing samples in file: {} (might take some time) + Saving model at: {}'.format(
                dump_file, model_save_path))
            # saver_.save(sess, model_save_path, global_step)
            t_ = '\n================= EPOCH {} =================\n'.format(
                eidx)
            # ys_samples, xs_samples = sess.run([ys[2], xs[2]])
            # print('####', ys_samples[:2])
            # print('####',xs_samples[:2])
            samples, samples_score, target_strs, source_strs = sess.run(
                [finished_seq, finished_scores, ys[2], xs[2]])

            # print('samples: {}\nsamples_score: {}\ntarget_strs: {}\nsource_strs: {}'.format(
            #     samples.shape, samples_score.shape, target_strs.shape, source_strs.shape
            # ))

            samples = samples.astype(np.int32).tolist()
            pred_strings = [[spm_encoder.decode_ids(
                samp_) for samp_ in samp] for samp in samples]
            # t_ += '{}\n'.format(pred_strings)
            for src_, trg_, pre_ in zip(source_strs, target_strs, pred_strings):
                t_ += '---- SOURCE (trunc-5h): {}\n'.format(
                    src_.decode('utf-8')[:500])
                t_ += '---- TARGET (trunc-5h): {}\n'.format(
                    trg_.decode('utf-8')[:500])
                for _pred_ in pre_:
                    t_ += '>>> PRED: {}\n'.format(_pred_)
                t_ += '\n'
            with open(dump_file, 'a', encoding='utf-8') as f:
                f.write(t_)
