import numpy as np


def normalize_word(word):
    return word.replace(":", "COL")


def normalize_label(label):
    norm_label = label.replace("``", "O_QUOT")\
        .replace("''", "C_QUOT")\
        .replace("'", "S_QUOT")\
        .replace(':', 'COL')\
        .replace('?', 'QMARK')

    if not len(norm_label):
        norm_label = "*EMPTY*"

    return norm_label


def opt_to_mean(o):
    """
    From bimu.
    :param o: npy object
    """
    if o.ndim == 3:
        return lambda x: np.mean(x, axis=0)  # avgEmb
    elif o.ndim == 2:
        return lambda x: x  # identity
    else:
        raise NotImplementedError


def contexts(words, i, vocab, win=3):

    win_start = max(0, i-win)
    win_end = min(len(words), i+win+1)

    cs = []
    for i_left in range(win_start, i):
        cs.append(vocab.get(words[i_left], -1))
    if i < win_end:
        for i_right in range(i, win_end):
            cs.append(vocab.get(words[i_right], -1))
    return cs


def softmax(x):
    assert x.ndim == 1
    return np.exp(x) / np.sum(np.exp(x))


def wordreps(words, cpost_tags, i, embs, vocab, c_embs=None):
    feats = []
    word = words[i]
    w_id = vocab.get(word, -1)

    if w_id != -1:
        if embs.ndim == 3:
            if c_embs is not None:  # try avgExp
                assert c_embs.ndim == 2
                cs = contexts(words, i, vocab)
                if cs:
                    cmean = np.mean(c_embs[np.array(cs)], axis=0)
                    act = np.dot(embs[w_id], cmean)
                    emb = np.average(embs[w_id], weights=softmax(act), axis=0)
                else:
                    emb = np.mean(embs[w_id], axis=0)
            else:
                emb = np.mean(embs[w_id], axis=0)
        else:
            emb = embs[w_id]

        for j, v in enumerate(emb):
            feats.append("emb{}={}".format(j, round(v, 3)))

    return feats


def taskar12(words, cpos_tags, i):
    # Features from http://www.seas.upenn.edu/~taskar/pubs/wikipos_emnlp12.pdf,
    # except the frequency filters are not applied.

    #  Word identity - lowercased word form if the word appears more than 10 times in the corpus.
    #  Hyphen - word contains a hyphen
    #  Capital - word is uppercased
    #  Suffix - last 2 and 3 letters of a word if they appear in more than 20 different word types.
    feats = []
    word = words[i]
    norm = normalize_word(word.lower())
    feats.append("w={}".format(norm))
    feats.append("suf2={}".format(norm[-2:]))
    feats.append("suf3={}".format(norm[-3:]))

    if word[0].isupper():
        feats.append("uppercased")

    if any(map(str.isdigit, norm)):
        feats.append("hasdigit")

    return feats


def honnibal13(words, cpos_tags, i):
    # Features from http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/.
    #
    # Features involving tags are skipped, since they are specific to the sequential prediction approach of that tagger.
    feats = []
    def add(name, *args):
        feats.append(u'='.join((name,) + tuple(args)))

    norm = normalize_word(words[i].lower())
    add('w', norm)
    add('pref1', norm[0])
    add('suf3', norm[-3:])

    if i > 0:
        prev = normalize_word(words[i-1].lower())
        add('<w', prev)
        add('<suf3', prev[-3:])

    if i > 1:
        prev2 = normalize_word(words[i-2].lower())
        add('<<w', prev2)

    if i < (len(words)-1):
        next = normalize_word(words[i+1].lower())
        add('>w', next)
        add('>suf3', next[-3:])
    #
    if i < (len(words)-2):
        next2 = normalize_word(words[i+2].lower())
        add('>>w', next2)

    return feats


def honnibal13_groups(words, cpos_tags, i):
    feats = []
    def add(name, group, *args):
        feat_str = u"{}={}".format(name, u"+".join(args))
        if group:
            feat_str += u"@{}".format(group)

        feats.append(feat_str)

    norm = normalize_word(words[i].lower())
    add('w', norm, norm)
    add('pref1', None, norm[0])
    add('suf3', None, norm[-3:])

    if i > 0:
        prev = normalize_word(words[i-1].lower())
        add('<w', prev, prev)
        add('<suf3', None, prev[-3:])

    if i > 1:
        prev2 = normalize_word(words[i-2].lower())
        add('<<w', prev2, prev2)

    if i < (len(words)-1):
        next = normalize_word(words[i+1].lower())
        add('>w', next, next)
        add('>suf3', None, next[-3:])
    #
    if i < (len(words)-2):
        next2 = normalize_word(words[i+2].lower())
        add('>>w', next2, next2)

    return feats
