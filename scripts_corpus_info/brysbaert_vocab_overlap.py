import argparse
import logging
import os
import sys

from ..core.classes import CorpusMetaData


logger = logging.getLogger()


def main(corpus_path, output_dir):

    wordlist_meta = CorpusMetaData(
        name="Brysbaert 1 word", path="/Users/caiwingfield/corpora/brysbaert40k/brysbaert1.wordlist")

    token_delimiter = "\n"

    info_filename = "Brysbaert 1-word vocab overlap.info"

    logger.info(f"Loading wordlist from {wordlist_meta.path}")
    with open(wordlist_meta.path, mode="r", encoding="utf-8") as wordlist_file:
        vocab_wordlist = set(wordlist_file.read().split(token_delimiter))

    logger.info(f"Wordlist has a vocab of size {len(vocab_wordlist):,}")

    logger.info(f"Loading corpus documents from {corpus_path}")
    with open(corpus_path, mode="r", encoding="utf-8") as corpus_file:
        vocab_corpus = set()
        for i, line in enumerate(corpus_file):
            vocab_corpus.add(line.strip())
            if i % 10_000_000 == 0:
                logger.info(f"\tRead {i:,} tokens")

    with open(os.path.join(output_dir, info_filename), mode="w", encoding="utf-8") as info_file:
        message = f"Corpus has a vocab of size {len(vocab_corpus):,}"
        log_and_write(message, info_file)

        overlap_vocab = set.intersection(vocab_corpus, vocab_wordlist)

        message = f"Overlap has a size of\t{len(overlap_vocab):,}"
        log_and_write(message, info_file)

        missing_vocab = vocab_wordlist - vocab_corpus

        message = f"Missing words: {len(missing_vocab)}"
        log_and_write(message, info_file)

        message = f"{missing_vocab}"
        log_and_write(message, info_file)


def log_and_write(message, info_file):
    logger.info(message)
    info_file.write(message + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpuspath")
    parser.add_argument("--outdir")
    args = vars(parser.parse_args())

    main(corpus_path=args["corpuspath"], output_dir=args["outdir"])

    logger.info("Done!")
