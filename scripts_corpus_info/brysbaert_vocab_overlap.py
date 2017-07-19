import logging
import os
import sys

from nltk.corpus import PlaintextCorpusReader

from ..core.classes import CorpusMetaData

logger = logging.getLogger()


def main():

    corpus_metas = [
        CorpusMetaData(
            name="BBC", path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus"),
        CorpusMetaData(
            name="BNC", path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus"),
    ]

    logger.info(f"Loading wordlist from /Users/caiwingfield/corpora/brysbaert40k/brysbaert1.txt")
    wordlist = PlaintextCorpusReader("/Users/caiwingfield/corpora/brysbaert40k/", "brysbaert1.txt").raw().split("\n")
    vocab_wordlist = set(wordlist)

    logger.info(f"Wordlist has a vocab of size {len(vocab_wordlist):,}")

    for corpus_meta in corpus_metas:

        logger.info(f"Loading {corpus_meta.name} corpus documents from {corpus_meta.path}")
        corpus_dir, corpus_filename = os.path.split(corpus_meta.path)
        corpus = PlaintextCorpusReader(corpus_dir, corpus_filename).raw().split("\n")
        vocab_corpus = set(corpus)
        logger.info(f"Corpus has a vocab of size {len(vocab_corpus):,}")

        overlap_vocab = set.intersection(vocab_corpus, vocab_wordlist)
        logger.info(f"Overlap with cutoff freq {cutoff_freq} has a size of\t{len(overlap_vocab):,}")

        missing_vocab = vocab_wordlist - vocab_corpus
        logger.info(f"Missing words: {len(missing_vocab)}")
        logger.info(f"{missing_vocab}")


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename="/Users/caiwingfield/Dropbox/brysbaert.log")
    logger.info("Running %s" % " ".join(sys.argv))
    logger.info("")

    main()

    logger.info("Done!")
