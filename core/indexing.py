def index_dictionary(freq_dist):
    """
    Builds dictionaries of word indices (starting at 0).  Indices will be sorted by token frequency
    :param freq_dist: A nltk.probability.FreqDist.
    :return:
    """

    word2id = {}
    current_id = 0
    for token, _freq in freq_dist.most_common():
        word2id[token] = current_id
        current_id += 1

    id2word = dict((v, k) for k, v in word2id.items())

    return word2id, id2word
