import string
import re

import nltk
import nltk.tokenize as tokenize
import nltk.corpus as corpus

from cw_common import *


def main():
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/BBC/2 No metaspeech"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.srt")

    # tokens = [token
    #           for token in tokenize.word_tokenize(corpus_text.raw())
    #           if len(token) == 1]

    tokens = [token
              for token in tokenize.word_tokenize(corpus_text.raw())
              if re.fullmatch('[' + string.punctuation + ']+', token)]

    fd = nltk.probability.FreqDist(tokens)

    mf = fd.most_common()

    print(mf)


if __name__ == '__main__':
    prints("Starting")
    main()
    prints("Done")
