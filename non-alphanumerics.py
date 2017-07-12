from modified_treebank import *

import nltk.corpus as corpus

from cw_common import *


def main():
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/Pathology"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.txt")

    tokens = modified_word_tokenize(corpus_text.raw())

    print(tokens)


if __name__ == '__main__':
    prints("Starting")
    main()
    prints("Done")
