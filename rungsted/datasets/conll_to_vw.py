# coding: utf-8
"""Create Rungsted compatible feature files from treebanks

Usage:
  conll_to_vw.py <input> <output> [--feature-set NAME] [--name NAME] [--coarse] [--embs FILE] [--vocab FILE]
  conll_to_vw.py (-h | --help)

Options:
  -h --help                 Show this screen.
  --feature-set NAME        Which feature to use [default: honnibal13].
  --name NAME               Name of dataset. Output as part of the id for each token [default: d].
  --coarse                  Use coarse-grained tags.
  --embs NAME                    File to embeddings in npy.
  --vocab NAME                   File to embdding vocabulary in txt.
"""
import codecs
from collections import defaultdict
from docopt import docopt
import numpy as np
from pos_features import taskar12, honnibal13, honnibal13_groups, normalize_label, normalize_word, wordreps

if __name__ == '__main__':
    args = docopt(__doc__)

    data_out = codecs.open(args['<output>'], 'w', encoding='utf-8')
    data_in = codecs.open(args['<input>'], encoding='utf-8')

    features_for_token = {
        'taskar12': taskar12,
        'honnibal13': honnibal13,
        'honnibal13-groups': honnibal13_groups,
        'wordreps': wordreps
    }[args['--feature-set']]

    def output_sentence(sent, embs=None, vocab=None):
        label_key = 'cpos' if args['--coarse'] else 'pos'
        if embs is not None and vocab is not None:
            for i in range(len(sent['word'])):
                print >>data_out, "{label} '{name}-{sent_i}-{token_i}|".format(
                    label=normalize_label(sent[label_key][i]),
                    name=args['--name'],
                    sent_i=sent_i,
                    token_i=i+1),
                print >>data_out, u" ".join(features_for_token(sent['word'], sent[label_key], i, embs=embs, vocab=vocab))
        else:
            for i in range(len(sent['word'])):
                print >>data_out, "{label} '{name}-{sent_i}-{token_i}|".format(
                    label=normalize_label(sent[label_key][i]),
                    name=args['--name'],
                    sent_i=sent_i,
                    token_i=i+1),
                print >>data_out, u" ".join(features_for_token(sent['word'], sent[label_key], i))


    embs = None
    vocab = None
    if args['--feature-set'] == "wordreps":
        embs = np.load(args["--embs"])
        vocab_f = codecs.open(args['--vocab'], encoding='utf-8')
        vocab = {l.strip(): i for i, l in enumerate(vocab_f)}
        vocab_f.close()
    # Process one sentence at a time
    sent = defaultdict(list)
    sent_i = 1
    for line in data_in:
        parts = line.strip().split()
        if len(parts) == 10:
            word = parts[1]
            if word.isdigit():
                number = int(word)
                if 1800 <= number <= 2100:
                    word = "!YEAR"
                else:
                    word = "!DIGITS"

            sent['word'].append(word)
            sent['cpos'].append(parts[3])
            sent['pos'].append(parts[4])
        elif len(parts) == 0:
            if sent_i > 1:
                print >>data_out, ""
            output_sentence(sent, embs, vocab)
            sent_i += 1
            sent = defaultdict(list)
        else:
            raise "Invalid input format"

    if len(sent['word']):
        output_sentence(sent, embs, vocab)