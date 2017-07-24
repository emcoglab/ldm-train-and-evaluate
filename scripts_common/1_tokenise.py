import glob
import logging
import os
import sys

import nltk

from ..core.classes import SourceTargetPair, CorpusMetaData
from ..core.filtering import filter_punctuation, ignorable_punctuation
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

        source_paths = glob.glob(os.path.join(corpus_meta.source.path, "*.*"))
        source_filenames = [os.path.basename(path) for path in source_paths]

        for i_1, source_filename in enumerate(source_filenames, start=1):

            corpus = nltk.corpus.PlaintextCorpusReader(corpus_meta.source.path, source_filename).raw()

            corpus = modified_word_tokenize(corpus)

            logger.info(f"Done {i:,} files")

            # Filter punctuation
            corpus = filter_punctuation(corpus)

            # TODO: deal with case where this already exists when first running the script
            with open(corpus_meta.target.path, mode="a", encoding="utf-8") as corpus_file:
                for token in corpus:
                    corpus_file.write(token+token_delimiter)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
