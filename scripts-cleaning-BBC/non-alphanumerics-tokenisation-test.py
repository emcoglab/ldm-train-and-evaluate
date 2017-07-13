import os
import sys

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
    corpus_root = "/Users/caiwingfield/Langboot local/Corpora/Pathology"

    corpus_text = corpus.PlaintextCorpusReader(
        corpus_root, ".*\.txt")

    tokens = modified_word_tokenize(corpus_text.raw())

    print(tokens)


if __name__ == '__main__':
    prints("Starting")
    main()
    prints("Done")
