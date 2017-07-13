import string
import re
import os
import sys

import nltk
import nltk.corpus as corpus

# oh god python is garbage sometimes
# to import a module from a sister directory you have to do this
sys.path.append(os.path.abspath('../core'))
# But pycharm doesn't understand it and thinks the modules don't exist
# noinspection PyUnresolvedReferences
import modified_tokenizer
# noinspection PyUnresolvedReferences
import ignorable_punctuation


def main():
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/BBC/3 Replaced symbols"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.srt")

    tokens = [token
              # only tokens which...
              for token in modified_tokenizer.modified_word_tokenize(corpus_text.raw())
              # ...are only a single character long
              if len(token) == 1
              # # ...are just made of punctuation
              # or re.fullmatch('[' + string.punctuation + ']+', token)
              # # ...are not a word-with-boring-punctuation
              # or (not re.fullmatch('[A-Za-z0-9' + ignorable_punctuation.ignorable_punctuation + ']+', token))
              ]

    fd = nltk.probability.FreqDist(tokens)

    mf = fd.most_common()

    print(mf)


if __name__ == '__main__':
    prints("Starting")
    main()
    prints("Done")
