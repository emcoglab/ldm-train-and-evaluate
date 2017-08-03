import pickle

import nltk


class TokenIndexDictionary(object):

    def __init__(self, token2id, id2token):
        """
        Constructor.
        :param token2id:
        :param id2token:
        """
        # A dictionary of token-keyed indices
        self.token2id = token2id
        # A dictionary of index-keyed tokens
        self.id2token = id2token

    def __len__(self):
        """
        The number of indexed tokens in the dictionary
        :return:
        """
        return len(self.token2id)

    def save(self, filename):
        """
        Saves an TokenIndexDictionary to a file
        :param filename:
        :return:
        """
        with open(filename, mode="wb") as file:
            # TODO: Don't use pickle
            pickle.dump(self, file)

    @classmethod
    def from_freqdist(cls, freq_dist: nltk.probability.FreqDist) -> 'TokenIndexDictionary':
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

        return cls(word2id, id2word)

    @classmethod
    def load(cls, filename) -> 'TokenIndexDictionary':
        """
        Loads an TokenIndexDictionary from a file.
        :param filename:
        :return:
        """
        with open(filename, mode="rb") as file:
            return pickle.load(file)
