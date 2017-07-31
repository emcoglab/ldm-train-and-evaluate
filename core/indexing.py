import pickle

import nltk


class TokenIndexDictionary(object):

    # A dictionary of token-keyed indices
    token2id = None

    # A dictionary of index-keyed tokens
    id2token = None

    def __init__(self, word2id, id2word):
        """
        Constructor.
        :param word2id:
        :param id2word:
        """
        self.token2id = word2id
        self.id2token = id2word

    def save(self, filename):
        """
        Saves an TokenIndexDictionary to a file
        :param filename:
        :return:
        """
        with open(filename, mode="wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def from_freqdist(freq_dist: nltk.probability.FreqDist):
        """
        Constructs an TokenIndexDictionary from a nltk.probability.FreqDist
        :param freq_dist:
        :return:
        """
        word2id = {}
        current_id = 0
        for token, _freq in freq_dist.most_common():
            word2id[token] = current_id
            current_id += 1

        id2word = dict((v, k) for k, v in word2id.items())

        return TokenIndexDictionary(word2id, id2word)

    @staticmethod
    def load(filename):
        """
        Loads an TokenIndexDictionary from a file.
        :param filename:
        :return:
        """
        with open(filename, mode="rb") as file:
            return pickle.load(file)
