import glob
import logging
import os
import sys

import nltk

from ..core.classes import SourceTargetPair, CorpusMetaData
from ..core.filtering import filter_punctuation
from ..core.tokenising import modified_word_tokenize

logger = logging.getLogger()


def main():

    corpus_metas = [
        SourceTargetPair(
            source=CorpusMetaData(
                name="BBC",  path="/Users/caiwingfield/corpora/BBC/3 Replaced symbols"),
            target=CorpusMetaData(
                name="BBC", path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus")),
        SourceTargetPair(
            source=CorpusMetaData(
                name="BNC", path="/Users/caiwingfield/corpora/BNC/1 Detagged"),
            target=CorpusMetaData(
                name="BNC",  path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus")),
        SourceTargetPair(
            source=CorpusMetaData(
                name="UKWAC", path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/2 Partitioned"),
            target=CorpusMetaData(
                name="UKWAC", path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/3 Tokenised/UKWAC.corpus"))]

    token_delimiter = "\n"

    for corpus_meta in corpus_metas:

        logger.info(f"Loading {corpus_meta.source.name} corpus from {corpus_meta.source.path}")
        logger.info(f"Tokenising corpus")

        token_count = 0

        with open(corpus_meta.target.path, mode="w", encoding="utf-8") as tokenised_corpus_file:

            source_paths = glob.glob(os.path.join(corpus_meta.source.path, "*.*"))
            source_filenames = [os.path.basename(path) for path in source_paths]

            for source_filename in source_filenames:

                corpus = nltk.corpus.PlaintextCorpusReader(corpus_meta.source.path, source_filename).raw()

                corpus = modified_word_tokenize(corpus)

                # Filter punctuation
                corpus = filter_punctuation(corpus)

                for token in corpus:

                    tokenised_corpus_file.write(token + token_delimiter)
                    token_count += 1

                    if token_count % 1_000_000 == 0 and token_count > 0:
                        logger.info(f"\tWritten {i:,} tokens")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
