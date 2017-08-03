import json

import nltk


class TokenIndexDictionary(object):

    def __init__(self, token2id):
        """
        Constructor.
        :param token2id:
        """
        # A dictionary of token-keyed indices
        self.token2id = token2id
        # A dictionary of index-keyed tokens
        self.id2token = dict((v, k) for k, v in token2id.items())

    def __len__(self):
        """
        The number of indexed tokens in the dictionary
        :return:
        """
        return len(self.token2id)

    def save(self, filename):
        """
        Saves a TokenIndexDictionary to a file
        :param filename:
        :return:
        """
        with open(filename, mode="wb") as file:
            json.dump(self, file,
                      # Remove whitespace for smaller files
                      separators=(',', ':'))

    @classmethod
    def from_freqdist(cls, freq_dist: nltk.probability.FreqDist) -> 'TokenIndexDictionary':
        """
        Constructs an TokenIndexDictionary from a nltk.probability.FreqDist
        :param freq_dist:
        :return:
        """
        token2id = {}
        current_id = 0
        for token, _freq in freq_dist.most_common():
            token2id[token] = current_id
            current_id += 1

        return cls(token2id)

    @classmethod
    def load(cls, filename) -> 'TokenIndexDictionary':
        """
        Loads an TokenIndexDictionary from a file.
        :param filename:
        :return:
        """
        with open(filename, mode="rb") as file:
            return cls(json.load(file))
