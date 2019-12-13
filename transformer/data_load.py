# -*- coding: utf-8 -*-
# inspired from https://www.github.com/kyubyong/transformer

import tensorflow as tf
import sentencepiece as spm

class Encoder:
    def __init__(self, encoder_path):
        sp = spm.SentencePieceProcessor()
        sp.load(encoder_path)
        self.encoder = sp
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()
        self.pad_id = sp.pad_id()
        self.unk_id = sp.unk_id()
        # self.vocab_size = sp.

    def encode_as_ids(self, inp):
        return self.encoder.encode_as_ids(inp)

    def encode_as_pieces(self, inp):
        return self.encoder.encode_as_pieces(inp)

    def decode_ids(self, inp):
        # print('>>>>>', inp)
        return self.encoder.decode_ids(inp)

def calc_num_batches(total_num, batch_size):
    return total_num // batch_size + int(total_num % batch_size != 0)

def load_data(fpath):
    sents1, sents2 = [], []
    with open(fpath, 'r', encoding = 'utf-8') as f1, open(fpath.replace('en', 'fr'), 'r', encoding = 'utf-8') as f2:
        en_lines = f1.readlines()
        fr_lines = f2.readlines()
        assert len(en_lines) == len(fr_lines)
        for line_idx in range(len(en_lines)):
            sents1.append(en_lines[line_idx].strip())
            sents2.append(fr_lines[line_idx].strip())
    return sents1, sents2

def encode(inp, type, encoder):
    inp_str = inp.decode("utf-8")
    tokens = encoder.encode_as_ids(inp)
    if type!="x": tokens = [encoder.bos_id] + tokens
    return tokens + [encoder.eos_id]

def generator_fn(sents1, sents2, encoder_fpath, maxlen1, maxlen2):
    # token2idx, _ = load_vocab(vocab_fpath)
    encoder = Encoder(encoder_fpath)
    maxlen1, maxlen2 = int(maxlen1), int(maxlen2)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", encoder)[:maxlen1]
        y = encode(sent2, "y", encoder)[:maxlen2]
        # decoder_input, y = y[:-1], y[1:] # --> we have managed this in model
        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (y, y_seqlen, sent2)
        # yield (x, x_seqlen), (decoder_input, y, y_seqlen)

def input_fn(sents1, sents2, encoder_fpath, batch_size, maxlen1, maxlen2, shuffle=False):
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    enc_temp = Encoder(encoder_fpath)

    shapes = (([None], (), ()),
              ([None], (), ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.string))
    paddings = ((enc_temp.pad_id, 0, ''),
                (enc_temp.pad_id, 0, ''))

    del enc_temp # save memory

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, encoder_fpath, maxlen1, maxlen2))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(fpath, encoder_fpath, maxlen1, maxlen2, batch_size, shuffle=False):
    sents1, sents2 = load_data(fpath = fpath)
    batches = input_fn(sents1 = sents1, sents2 = sents2,
        encoder_fpath=encoder_fpath, batch_size = batch_size,
        maxlen1 = str(maxlen1), maxlen2 = str(maxlen2),
        shuffle=shuffle) # conversion to string helps in passing it as an argument
    num_batches = calc_num_batches(total_num= len(sents1), batch_size =batch_size)
    return batches, num_batches, len(sents1)
