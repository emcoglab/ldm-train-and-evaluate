import os
import re
import sys

import logging
import pickle

import gensim as gs
import nltk
import nltk.corpus

from modified_tokeniser import modified_word_tokenize, ignorable_punctuation

logger = logging.getLogger()


def main():

    unfiltered_corpus_dir = "/Users/caiwingfield/Langboot local/Corpora/BBC/3 Replaced symbols"
    filtered_corpus_dir   = "/Users/caiwingfield/Langboot local/Corpora/BBC/4 Filtered corpus"

    logger.info("Loading and tokenising corpus")

    corpus = modified_word_tokenize(
        nltk.corpus.PlaintextCorpusReader(
            unfiltered_corpus_dir, ".*\.srt"
        ).raw())

    words_freq_dist = nltk.probability.FreqDist(corpus)

    min_freq = 5

    logger.info("Filtering corpus based on token frequency")
    logger.info(f"Removing all tokens appearing fewer than {min_freq} times")
    logger.info(f"{len(corpus)} tokens in corpus before filtering")

    corpus = [token
              for token in corpus
              if not re.fullmatch('[' + ignorable_punctuation + ']+', token)
              and words_freq_dist[token] >= min_freq]

    logger.info(f"{len(corpus)} tokens remaining in corpus after filtering")

    logger.info("Building word index dictionaries")

    # We don't care about documents, so just include everything in one document
    id2word = gs.corpora.Dictionary([corpus])
    word2id = dict((v, k) for k, v in id2word.iteritems())
    id2word = gs.utils.revdict(word2id)

    with open(os.path.join(filtered_corpus_dir, "corpus.p"), mode="wb") as corpus_file:
        pickle.dump(corpus, corpus_file)

    with open(os.path.join(filtered_corpus_dir, "id2word.p"), mode="wb") as id2word_file:
        pickle.dump(id2word, id2word_file)

    with open(os.path.join(filtered_corpus_dir, "word2id.p"), mode="wb") as word2id_file:
        pickle.dump(word2id, word2id_file)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
