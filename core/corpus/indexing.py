"""
===========================
Converting string tokens to and from int indexes.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

import json
import os
import pickle
import logging

import nltk

from core.corpus.corpus import BatchedCorpus

logger = logging.getLogger(__name__)


# noinspection PyAbstractClass
class FreqDist(nltk.probability.FreqDist):
    """
    Extension of nltk.probability.FreqDist with a few useful helper methods.
    """

    _file_extension = ".freqdist"

    @classmethod
    def could_load(cls, filename: str) -> bool:
        if not filename.endswith(FreqDist._file_extension):
            filename += FreqDist._file_extension
        return os.path.isfile(filename)

    def save(self, filename: str):
        if not filename.endswith(FreqDist._file_extension):
            filename += FreqDist._file_extension
        with open(filename, mode="wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename: str) -> 'FreqDist':
        if not filename.endswith(FreqDist._file_extension):
            filename += FreqDist._file_extension
        with open(filename, mode="rb") as file:
            return pickle.load(file)

    @classmethod
    def from_batched_corpus(cls, batched_corpus: BatchedCorpus) -> 'FreqDist':
        # Initially create an empty FreqDist
        freq_dist = cls()
        # Then add to it from each batch of the corpus.
        # Means we don't have to have the whole corpus in memory at once!
        token_count = 0
        for batch in batched_corpus:
            freq_dist += cls(batch)
            token_count += batched_corpus.batch_size
            logger.info(f"\t{token_count:,} tokens counted")
        return freq_dist


class LetterIndexing(object):
    _alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                 "u", "v", "w", "x", "y", "z"]

    @staticmethod
    def letter2int(letter: str) -> int:
        """
        Converts roman letters a, b, c, ... into integers 0, 1, 2, ...
        """
        i = LetterIndexing._alphabet.index(letter)
        if i == -1:
            raise KeyError()
        return i

    @staticmethod
    def int2letter(i: int) -> str:
        """
        Converts integers 0, 1, 2, ... into roman letters a, b, c, ...
        """
        return LetterIndexing._alphabet[i]


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

    @property
    def tokens(self):
        return [k for k, v in self.token2id.items()]

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
        with open(filename, mode="w") as file:
            json.dump(self.token2id, file,
                      # Remove whitespace for smaller files
                      separators=(',', ':'))

    @classmethod
    def from_freqdist(cls, freq_dist: FreqDist) -> 'TokenIndexDictionary':
        """
        Constructs an TokenIndexDictionary from a FreqDist
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
        with open(filename, mode="r") as file:
            return cls(json.load(file))