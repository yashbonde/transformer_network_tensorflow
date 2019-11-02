# test file for building the model and testing out the functionalities

# import dependencies
import numpy as np
import argparse
import os
from network import TransformerNetwork

# args
parser = argparse.ArgumentParser()
parser.add_argument('--num-examples', type = int, default = 100, help = 'number of examples')
parser.add_argument('--qp-factor', type = int, default = 4, help = 'num passages per query')
parser.add_argument('--vocab-size', type = int, default = 300, help = 'size of vocabulary')
parser.add_argument('--emb-dim', type = int, default = 20, help = 'embedding dimension')
parser.add_argument('--save-folder', type = str, default = './test', help = 'path to folder where saving model')

# this is same for passages and queries
parser.add_argument('--seqlen', type = int, default = 40, help = 'length of sequences')
args = parser.parse_args()

# making toy data
print('[*] Arguments:')
print('--num-examples:', args.num_examples)
print('--qp-factor:', args.qp_factor)
print('--vocab-size:', args.vocab_size)
print('--emb-dim:', args.emb_dim)
print('--seqlen:', args.seqlen)

# save folder
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
    print('[!] Model saving folder could not be found, making folder ', args.save_folder)

# === Queries === #
len_q = np.random.randint(low = 2, high = args.seqlen - 4, size = (int(args.num_examples/args.qp_factor)))
queries_ = []
for q in len_q:
  query = [np.random.randint(args.vocab_size, size = (q))] * args.qp_factor
  queries_.extend(query)
queries_ = np.array(queries_)
del len_q # save memory
print('[*] queries_:', queries_.shape)

# === Passages === #
len_p = np.random.randint(low = 5, high = args.seqlen, size = (args.num_examples))
passage_ = np.array([np.random.randint(args.vocab_size, size = (p)) for p in len_p])
del len_p # save memory
print('[*] passage_:', passage_.shape)

# === labels === #
label_oh = np.random.randint(args.qp_factor, size = (int(args.num_examples/args.qp_factor)))
label_ = []
for i in label_oh:
  l = np.zeros(args.qp_factor, dtype = np.int32).tolist()
  l[i] = 1
  label_.extend(l)
del label_oh # save memory
label_ = np.array(label_)
print('[*] label_:', label_.shape)

# === embeddings === #
embedding_mat = np.random.random((args.vocab_size+1, args.emb_dim))

# build the network
model = TransformerNetwork(scope = 'net_test',
                           save_folder = args.save_folder,
                           pad_id = int(args.vocab_size),
                           save_freq = 1,
                           dim_model = args.emb_dim)
model.build_model(emb = embedding_mat, seqlen = args.seqlen - 2, print_stack = True)
# model.print_network()

# train the network
model.train(queries_ = queries_,
            passage_ = passage_,
            label_ = label_,
            num_epochs = 2,
            val_split = 0.2)

# https://stackoverflow.com/a/19747562
raise SystemExit # efficient way to exit program