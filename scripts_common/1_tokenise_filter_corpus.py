import os
import sys
import logging
import pickle

import nltk
import nltk.corpus

from ..core.tokenising import modified_word_tokenize
from ..core.filtering import filter_frequency, filter_punctuation, ignorable_punctuation
from ..core.indexing import index_dictionary

logger = logging.getLogger()


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

    min_freq = 0

    logger.info(f"Filtering corpus based on token frequency")
    logger.info(f"{len(corpus)} tokens in corpus before filtering")

    logger.info(f"Removing punctuation tokens ({ignorable_punctuation})")
    if min_freq > 0:
        logger.info(f"Removing all tokens appearing fewer than {min_freq} times")

    freq_dist = nltk.probability.FreqDist(corpus)

    corpus = filter_punctuation(filter_frequency(corpus, min_freq=min_freq, freq_dist=freq_dist))

    logger.info(f"{len(corpus)} tokens remaining in corpus after filtering")

    logger.info("Building word index dictionaries")

    # We don't care about documents, so just include everything in one document
    word2id, id2word = index_dictionary(corpus, freq_dist=freq_dist)

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
