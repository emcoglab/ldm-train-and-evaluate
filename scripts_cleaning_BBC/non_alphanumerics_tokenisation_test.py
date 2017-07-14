import logging
import sys

import nltk.corpus as corpus

from ..core.modified_tokenizer import modified_word_tokenize

logger = logging.getLogger()


def main():
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/Pathology"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.txt")

    tokens = modified_word_tokenize(corpus_text.raw())

    print(tokens)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
