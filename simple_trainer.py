"""
trainer.py

script to train any model

24.09.2019 - @yashbonde
"""

import json
import logging
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from tqdm import tqdm

# custom
import transformer.baseline as transformer_model
from transformer.ops_util import ModelConfig
from transformer.beam_search import beam_search
from transformer.tf_layers import positions_for

config = ModelConfig(description = 'Script to perform complete experiment cycle for one model')
# config.add_arg('--arch', type = str, default = 'baseline', help = 'model architecture. For refrence goto README.md')

config.add_arg('--mode', type = int, default = 0, help = 'set 1 for complete test, 0 otherwise')

config.parse_args()
config.add_value('num_layers', 2)
config.add_value('num_heads', 2)
config.add_value('embedding_dim', 16)
config.add_value('vocab_size', 15)
config.add_value('cntx_len', 12)
config.add_value('model_save_path', './bubble')
config.add_value('use_inverse_embedding', False)
config.add_value('opt', 'adam')
config.add_value('lr', 0.01)
config.add_value('batch_size', 5)
config.add_value('beam_size', 7)
config.add_value('max_decode_length', 17)
config.add_value('length_penalty_factor', 2)
config.add_value('max_encoder_length', config.cntx_len)

if config.mode:
    num_samples = 23
    # load sample data and language
    lang1 = []
    lang2 = []
    for _ in range(config.batch_size):
        # language 1
        sent = (np.random.randint(config.vocab_size - 1,
            size = np.random.randint(high = config.max_encoder_length,
                                     low = 6)) + 1).tolist() 
        sent += [0,] * (config.max_encoder_length - len(sent)  - 1)
        lang1.append(sent)
        # language 2
        # plus 1 compensation for target language
        sent = (np.random.randint(config.vocab_size - 1,
            size = np.random.randint(high = config.max_decode_length,
                                     low = 12)) + 1).tolist()
        sent += [0,] * (config.max_decode_length - len(sent))
        lang2.append(sent)

    lang1 = np.array(lang1)
    lang2 = np.array(lang2) 

    print('----- LANGAUGE -----')
    print(lang1.shape, lang2.shape)
    for l1, l2 in zip(lang1, lang2):
        print(l1, l2)
        print(len(l1), len(l2))
    print('--------------------')

"""loading the model ---> using data generator now"""
# define placeholders
encoder_placeholder = tf.placeholder(tf.int32, [None, None], name = 'encoder_placeholder')
target_placeholder = tf.placeholder(tf.int32, [None, None], name = 'target_placeholder')

# training --> make the output sequence
model_namespace = transformer_model.transformer(
    config = config,
    encoder_placeholder = encoder_placeholder,
    target_placeholder = target_placeholder,
    training=True)

print('============== TRAINING ==============')
for key, value in model_namespace.__dict__.items():
    if key is not "train_step":
        print('****', key, '\n----', value)

# now we make one for beam decoding
def symbols_to_logits_fn(model, config, decoder_tensor):
    '''We basically need to run the complete decoder function
    model: namespace returned from function
    decoder_tensor: [batch_size * beam_size, decoded_length]
    :return new_ids: [batch_size * beam_size, vocab_size]
    '''
    print(f'^^^^^^ decoder_tensor: {decoder_tensor}')
    # gather and get the emebddings for the yet decoded sequence and send to the decoder
    decoder_gather = tf.gather(model.context_embedding, decoder_tensor) + \
        tf.gather(model.position_embedding, positions_for(decoder_tensor, past_length = 0))
    print(f'>>>>> {decoder_gather}')
    encoder_tiled = tf.tile(model.encoder_embedding, [config.beam_size, 1, 1])
    print(f'>>> encoder_tiled: {encoder_tiled}')
    decoder_out = transformer_model.decoder_fn(config, decoder_gather, encoder_tiled) # [bs, None, embedding_dim]
    decoder_out = tf.matmul(decoder_out, model.context_embedding, transpose_b = True) # [bs, None, vocab_size]
    print(f'>>>> decoder_out: {decoder_out}')
    decoder_out_last_step = decoder_out[:,-1,:]
    print(f'>>> decoder_out_last_step: {decoder_out_last_step}')
    # return tf.squeeze(decoder_out_last_step, axis = [1])
    return decoder_out_last_step


# if not config.mode:
print('============== TESTING ==============')
initial_ids = tf.constant([[3],] * config.batch_size, dtype = tf.int32)
print('*** Sample Sequence before wrapper:', initial_ids)
sample_sequence = symbols_to_logits_fn(model_namespace, config, initial_ids)
print('*** Sample Sequence from symbols_to_logits_fn wrapper:', sample_sequence)

initial_ids_loop = tf.squeeze(initial_ids, axis = [1])
print('****** initial_ids_loop', initial_ids_loop)

finished_seq, finished_scores, state = beam_search(config,
                model_namespace = model_namespace,
                symbols_to_logits_fn = symbols_to_logits_fn,
                initial_ids = initial_ids_loop,
                initial_id = None,
                beam_size = config.beam_size,
                decode_length = config.max_decode_length,
                vocab_size = config.vocab_size,
                alpha = config.length_penalty_factor,
                eos_id=2,
                states=None,
                stop_early=True)

print('========== DONE ===========')
print('****** finished_seq:', finished_seq) # [bs, max_decode_length, vocab_size]
print('****** finished_scores:', finished_scores) # [bs, max_decode_length]

if not config.mode:
    exit()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('\n----- LOSS TRAINING -----')
    for eidx in range(5): # train for 5 epochs
        ops = [model_namespace.loss, model_namespace.train_step, model_namespace.encoder_embedding]
        fd = {
            encoder_placeholder: lang1,
            target_placeholder: lang2
        }
        _loss, _, enc_emb = sess.run(ops, fd)
        print('>>> EPOCH: {}, loss: {}, shapes {}, {}, tensor_shape: {}'.format(
            eidx, _loss, lang1.shape, lang2.shape, enc_emb.shape))
    print('\n----- DECODE -----')
    for eidx in range(2): # now generate for 2 epochs
        fd = {encoder_placeholder: lang1}
        _fseq, _fscores = sess.run([finished_seq, finished_scores], fd)
        print('>>> EPOCH: {}\nseq: {}\nscores: {}'.format(eidx, _fseq, _fscores))

