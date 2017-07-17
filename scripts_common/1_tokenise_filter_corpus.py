import os
import re
import sys
import logging
import pickle

import nltk
import nltk.corpus

from ..core.modified_tokenizer import modified_word_tokenize
from ..core.ignorable_punctuation import ignorable_punctuation

logger = logging.getLogger()


def index_dictionary(corpus):
    """
    Returns a dictionary of
    :param corpus:
    :return:
    """
    vocab = sorted(set(corpus))

    word2id = {}
    count = 0
    for word in vocab:
        word2id[word] = count
        count += 1

    id2word = dict((v, k) for k, v in word2id.items())

    return word2id, id2word


def main():
    unfiltered_corpus_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/3 Replaced symbols"
    filtered_corpus_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/4 Tokenised and filtered"
    # unfiltered_corpus_dir = "/Users/caiwingfield/Langboot local/Corpora/toy-corpus/0 Raw"
    # filtered_corpus_dir   = "/Users/caiwingfield/Langboot local/Corpora/toy-corpus/1 Tokenised and filtered"

    logger.info("Loading and tokenising corpus")

    corpus = modified_word_tokenize(
        nltk.corpus.PlaintextCorpusReader(
            # Any file with name and extension
            unfiltered_corpus_dir, ".+\..+"
        ).raw())

    words_freq_dist = nltk.probability.FreqDist(corpus)

    min_freq = 0

    logger.info(f"Filtering corpus based on token frequency")
    logger.info(f"{len(corpus)} tokens in corpus before filtering")

    logger.info(f"Removing punctuation tokens ({ignorable_punctuation})")
    if min_freq > 0:
        logger.info(f"Removing all tokens appearing fewer than {min_freq} times")

    corpus = [token
              for token in corpus
              if not re.fullmatch('[' + ignorable_punctuation + ']+', token)
              and words_freq_dist[token] >= min_freq]

    logger.info(f"{len(corpus)} tokens remaining in corpus after filtering")

    logger.info("Building word index dictionaries")

    # We don't care about documents, so just include everything in one document
    word2id, id2word = index_dictionary(corpus)

    with open(os.path.join(filtered_corpus_dir, "corpus.p"), mode="wb") as corpus_file:
        pickle.dump(corpus, corpus_file)

    with open(os.path.join(filtered_corpus_dir, "id2word.p"), mode="wb") as id2word_file:
        pickle.dump(id2word, id2word_file)

    with open(os.path.join(filtered_corpus_dir, "word2id.p"), mode="wb") as word2id_file:
        pickle.dump(word2id, word2id_file)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
