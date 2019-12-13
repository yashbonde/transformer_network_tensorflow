"""
clean_debates.py
"""

import logging
import json
import sentencepiece as spm
import argparse
import os
from tqdm import tqdm
import regex as re

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description = 'Script to clean the downloaded NIPS papers data. The behaviour is slightly '
                                                  'unpredictable and depends upon your vocabulary.')
    parse.add_argument('--folder', default = './training', help = 'Folder with data, we will "walk" over this folder')
    parse.add_argument('--model', default='debates', help='Name of sentencepiece model')
    parse.add_argument('--sentence_length', default = 3000, type = int, help = 'Number of characters to keep in each sequence')
    parse.add_argument('--hard_vocab_limit', default = False, type = bool,
                       help = 'If the text cannot be split into a given vocab size the sentence piece returns an error,'
                              ' this is used to fix that problem. But this can result in dynamic sized vocabulary.')
    parse.add_argument('--vocab_size', default= 800, type = int, help = 'Size of the vocabulary to use')
    args = parse.parse_args()

    # find all jsons which have data
    pairs = []
    for root_dir, _, path in os.walk(args.folder):
        for p in path:
            if p.split('.')[-1] == 'e':
                en_path = os.path.join(root_dir, p)
                fr_path = en_path[:-2] +  '.f'

                if '_h' in en_path or '_h' in fr_path:
                    print('* Skipping: {}'.format(en_path, fr_path))
                    continue

                pairs.append((en_path, fr_path))

    english_paths = [p[0] for p in pairs]
    french_paths = [p[1] for p in pairs]

    # print(english_paths)

    # format
    re_pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    english_txt = []
    for i in tqdm(range(len(pairs))):
        try:
            curr_txt = open(english_paths[i], 'r', encoding = 'latin-1').read()
        except Exception as e:
            print('* Failed on file:', english_paths[i])
            print(e)
            continue
        t_ = ''.join(re.findall(re_pat, curr_txt))
        t_ = t_.replace('\n', '')[:args.sentence_length]
        english_txt.append(t_)

    print('****', len(english_txt[0]))

    french_text = []
    for i in tqdm(range(len(french_paths))):
        try:
            curr_txt = open(french_paths[i], 'r', encoding = 'latin-1').read()
        except:
            print('* Failed on file:', french_paths[i])
            continue
        t_ = ''.join(re.findall(re_pat, curr_txt))
        t_ = t_.replace('\n', '')[:args.sentence_length]
        french_text.append(t_)

    print('****', len(french_text[0]))

    # combine all to single text block
    all_tt = '\n'.join(english_txt + french_text)
    master_name_file = os.path.join(os.getcwd(), 'all_text_joined.txt')
    logging.warning('Writing output file to {}'.format('all_text_joined.txt'))
    with open(master_name_file, 'w', encoding='utf-8') as f:
        f.write(all_tt)

    # write english and french files separately
    all_tt = '\n'.join(english_txt)
    name_file = os.path.join(os.getcwd(), 'en.txt')
    logging.warning('Writing output file to {}'.format('en.txt'))
    with open(name_file, 'w', encoding='utf-8') as f:
        f.write(all_tt)

    all_tt = '\n'.join(french_text)
    name_file = os.path.join(os.getcwd(), 'fr.txt')
    logging.warning('Writing output file to {}'.format('fr.txt'))
    with open(name_file, 'w', encoding='utf-8') as f:
        f.write(all_tt)

    # make sentencepiece model
    hv = 'true' if args.hard_vocab_limit else 'false'
    spm.SentencePieceTrainer.train('--input={} \
                                    --model_prefix={} \
                                    --vocab_size={} \
                                    --normalization_rule_name=nmt_nfkc_cf\
                                    --hard_vocab_limit={}\
                                    --max_sentence_length={}\
                                    --pad_id=0 --unk_id=1\
                                    --bos_id=2 --eos_id=3'.format(master_name_file,
                                                                     args.model,
                                                                     args.vocab_size,
                                                                     hv,
                                                                     args.sentence_length * 3
                                                                  ))



