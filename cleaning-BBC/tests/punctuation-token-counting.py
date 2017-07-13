import string
import re

import nltk
import nltk.corpus as corpus

from cw_common import *
from core.modified_tokeniser import modified_word_tokenize


def main():
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/BBC/3 Replaced symbols"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.srt")

    tokens = [token
              # only tokens which...
              for token in modified_word_tokenize(corpus_text.raw())
              # ...are only a single character long
              if len(token) == 1
              # # ...are just made of punctuation
              # or re.fullmatch('[' + string.punctuation + ']+', token)
              # # ...are not a word-with-boring-punctuation
              # or (not re.fullmatch('[A-Za-z0-9' + string.punctuation + ']+', token))
              ]

    fd = nltk.probability.FreqDist(tokens)

    mf = fd.most_common()

    print(mf)


if __name__ == '__main__':
    prints("Starting")
    main()
    prints("Done")
