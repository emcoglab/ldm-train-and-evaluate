import logging
import sys

from ..core.classes import CorpusMetaData

logger = logging.getLogger()


def main():

    corpus_metas = [
        CorpusMetaData(
            name="BBC", path="/Users/cai/Desktop/BBC.corpus"),
        CorpusMetaData(
            name="BNC", path="/Users/cai/Desktop/BNC.corpus"),
    ]

    logger.info(f"Loading wordlist from /Users/cai/Desktop/b.txt")
    with open("/Users/cai/Desktop/b.txt", mode="r") as wordlist_file:
        vocab_wordlist = set(wordlist_file.read().split("\n"))

    logger.info(f"Wordlist has a vocab of size {len(vocab_wordlist):,}")

    for corpus_meta in corpus_metas:

        logger.info(f"Loading {corpus_meta.name} corpus documents from {corpus_meta.path}")
        with open(corpus_meta.path, mode="r") as corpus_file:
            vocab_corpus = set(corpus_file.read().split("\n"))
        logger.info(f"Corpus has a vocab of size {len(vocab_corpus):,}")

        overlap_vocab = set.intersection(vocab_corpus, vocab_wordlist)
        logger.info(f"Overlap has a size of\t{len(overlap_vocab):,}")

        missing_vocab = vocab_wordlist - vocab_corpus
        logger.info(f"Missing words: {len(missing_vocab)}")
        logger.info(f"{missing_vocab}")


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    logger.info("")

    main()

    logger.info("Done!")
