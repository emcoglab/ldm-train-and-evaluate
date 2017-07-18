import re

import nltk

from .modified_tokenizer import modified_word_tokenize
from .ignorable_punctuation import ignorable_punctuation


def filter_punct(unfiltered_corpus):
    """
    Filters a corpus by ignoring certain punctuation.
    :param unfiltered_corpus: A list of tokens, for example that provided by:
        nltk.corpus.PlaintextCorpusReader(unfiltered_corpus_dir, ".+\..+").raw()
    :return:
    """
    return [token
            for token in unfiltered_corpus
            if not re.fullmatch('[' + ignorable_punctuation + ']+', token)]


def filter_frequency(unfiltered_corpus, min_freq=0, freq_dist=None):
    """
    Filters a corpus by ignoring words which are too rare.
    :param unfiltered_corpus: A list of tokens, for example that provided by:
        nltk.corpus.PlaintextCorpusReader(unfiltered_corpus_dir, ".+\..+").raw()
    :param min_freq: Ignore any tokens which appear fewer times than this. Set to 0 to include all tokens.
    :param freq_dist: Optionally supply an existing frequency distribution to avoid re-computing it
    :return:
    """
    if min_freq is 0:
        return unfiltered_corpus
    else:
        if freq_dist is None:
            freq_dist = nltk.probability.FreqDist(unfiltered_corpus)
        return [token
                for token in unfiltered_corpus
                if freq_dist[token] >= min_freq]
