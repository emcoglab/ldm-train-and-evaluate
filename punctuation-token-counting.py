import string
import re

import nltk
import nltk.corpus as corpus

from cw_common import *
from modified_treebank import modified_word_tokenize


def main():
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/BBC/2 No metaspeech"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.srt")

    tokens = [token
              for token in modified_word_tokenize(corpus_text.raw())
              # If it's just made of punctuation
              if re.fullmatch('[' + string.punctuation + ']+', token)
              # Not a word-with-punctuation
              or not re.fullmatch('[^A-Za-z0-9' + string.punctuation + ']+', token)
              # only a single character long
              or len(token) == 1]

    fd = nltk.probability.FreqDist(tokens)

    mf = fd.most_common()

    print(mf)


if __name__ == '__main__':
    prints("Starting")
    main()
    prints("Done")
