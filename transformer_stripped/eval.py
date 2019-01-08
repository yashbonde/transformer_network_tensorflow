'''
Evaluation file for various models.

Aim of this file is to be as generic as possible so training any model is a breeze
all the major functions are already present here and the only thing that should can
be changed is the network file.
'''

# dependencies
import argparse
import numpy as np
import os
from glob import glob

# custom model
import network
from utils import add_padding

# batch iterator
def get_batch(iteratable, n = 20):
    iter_len = len(iteratable)
    for ndx in range(0, iter_len, n):
        temp = iteratable[ndx:min(ndx + n, iter_len)]
        if len(temp) < n:
            temp = iteratable[ndx:min(ndx + n, iter_len)].copy()
            gap = n - len(temp)
            temp.extend(iteratable[:gap])
        yield temp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qnqp-file', type = str, help = 'path to numpy dumps')
    parser.add_argument('--embedding-path', type = str, help = 'path to embedding .npy file')
    parser.add_argument('--model-path', type = str, help = 'path to folder where saving model')
    parser.add_argument('--results-path', type = str, help = 'path to folder where saving results')
    parser.add_argument('--model-name', type = str, default = 'final1', help = 'scope name')
    parser.add_argument('--batch-size', type = int, default = 1024, help = 'size of minibatch')
    parser.add_argument('--seqlen', type = int, default = 80, help = 'length of sequences')
    args = parser.parse_args()

    '''
    Step 1: Before the models is built and setup, do the preliminary work
    '''

    # make the folders
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        print('[!] Model saving folder could not be found, making folder, {args.results_path}')


    # We need to get the list of all the q, p and l files that were generated
    qn_paths = sorted(glob(args.qnqp_file + '_n*.npy'))
    q_paths = sorted(glob(args.qnqp_file + '_q*.npy'))
    p_paths = sorted(glob(args.qnqp_file + '_p*.npy'))

    print(qn_paths)
    print(q_paths)
    print(p_paths)

    '''
    Step 2: All the checks are done, make the model
    '''

    print('[*] Data loading ...')
    # load the training numpy matrix
    for i in range(len(q_paths)):
        print('... loading file number:', i)
        if i == 0:
            eval_qn = np.load(qn_paths[i])
            eval_q = np.load(q_paths[i])
            eval_p = np.load(p_paths[i])
        else:
            q_ = np.load(qn_paths[i])
            p_ = np.load(q_paths[i])
            l_ = np.load(p_paths[i])
            eval_qn = np.concatenate([eval_qn, q_])
            eval_q = np.concatenate([eval_q, p_])
            eval_p = np.concatenate([eval_p, l_])
    
    # load embedding matrix
    print('... loading embedding matrix')
    embedding_matrix = np.load(args.embedding_path)

    print('[*] ... Data loading complete!')

    # load the model, this is one line that will be changed for each case
    print('[*] Making model')
    model = network.TransformerNetwork(scope = args.model_name,
                                       save_folder = args.model_path,
                                       pad_id = len(embedding_matrix),
                                       is_training = False,
                                       dim_model = embedding_matrix.shape[-1])

    # build the model
    print('[*] Building model (for details look at the stack below)')
    model.build_model(emb = embedding_matrix,
                      seqlen = args.seqlen,
                      batch_size = args.batch_size,
                      print_stack = True)

    print('[*] Loading the model from saved weights')
    model.start_sess_loader()

    '''
    Step 3: get results from model
    '''
    # make an index range
    idx_ = np.arange(len(qn_)).tolist()

    # make list of results
    all_preds = []

    num_batches = 0
    proc_batches = 0

    for _ in get_batch(idx_, n = 512):
        num_batches += 1
        
    print('[!] number of batches:', num_batches)

    # iterate
    for x in get_batch(idx_, n = 512):
        proc_batches += 1
        q = add_padding(queries_[x], model.pad_id, 80)
        p = add_padding(passages_[x], model.pad_id, 80)
        
        feed_dict = {model.query_input: q, model.passage_input: p}
        preds = model.sess.run(model.net_op, feed_dict = feed_dict)
        all_preds.append(preds)

    '''
    Step 4: convert to proper list and write the answers.tsv file
    '''
    all_pred_ = np.reshape(all_preds, [-1])
    all_pred_ = all_pred_[:len(qn_)]
    all_pred_ = np.reshape(all_pred_, [-1, 10])

    # converting to the required dump format
    res_path = '../final1_saves/res1/final_26.tsv'

    # first convert the entirity to text
    text = ''
    for i in range(len(all_pred_)):
        tt = str(int(qn_[i*10]))
        for p in all_pred_[i]:
            tt += '\t' + ("%.6f" % p)
        tt += '\n'
        text += tt
        
    with open(res_path, 'w') as f:
        f.write(text)

    print('... Done! File saved at:', res_path)

