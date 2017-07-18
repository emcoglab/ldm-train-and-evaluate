def index_dictionary(corpus, freq_dist=None):
    """
    Builds dictionaries of word indices (starting at 0).
    :param corpus:
    :param freq_dist: Optionally supply a nltk.probability.FreqDist, in which case terms will be indexed in frequency
    order, else they will be indexed in alphabetical order.
    :return:
    """

    # If using a frequency distribution to index tokens in frequency order
    if freq_dist is not None:
        word2id = {}
        current_id = 0
        for token, _freq in freq_dist.most_common():
            word2id[token] = current_id
            current_id += 1

    # If indexing tokens in alphabetical order
    else:
        vocab = sorted(set(corpus))

        word2id = {}
        current_id = 0
        for token in vocab:
            word2id[token] = current_id
            current_id += 1

    id2word = dict((v, k) for k, v in word2id.items())

    return word2id, id2word
