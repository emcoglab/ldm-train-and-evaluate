import re

import nltk


def filter_punctuation(unfiltered_corpus):
    """
    Filters a corpus by ignoring certain punctuation.
    :param unfiltered_corpus: A list of tokens, for example that provided by:
        nltk.corpus.PlaintextCorpusReader(unfiltered_corpus_dir, ".+\..+").raw()
    :return:
    """
    return [token
            for token in unfiltered_corpus
            if not re.fullmatch('[' + ignorable_punctuation + ']+', token)]


def filter_frequency(unfiltered_corpus, min_freq=1, freq_dist=None):
    """
    Filters a corpus by ignoring words which are too rare.
    :param unfiltered_corpus: A list of tokens, for example that provided by:
        nltk.corpus.PlaintextCorpusReader(unfiltered_corpus_dir, ".+\..+").raw()
    :param min_freq: Ignore any tokens which appear fewer times than this. Set to 0 to include all tokens.
    :param freq_dist: Optionally supply an existing frequency distribution to avoid re-computing it
    :return:
    """
    if min_freq is 1:
        return unfiltered_corpus
    else:
        if freq_dist is None:
            freq_dist = nltk.probability.FreqDist(unfiltered_corpus)
        return [token
                for token in unfiltered_corpus
                if freq_dist[token] >= min_freq]

# string.punctuation  = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# same as above except:
#  - we don't want to ignore:
#       $
#       %
#       &
#       @
#       +
#       =
#  - we do want to ignore
#       … ellipsis
#       – en-dash
#       — em-dash
#       ‘ open single quote
#       ’ close single quote
ignorable_punctuation = r"""!"#'()*,-./:;<>?[\]^_`{|}~…–—‘’"""
