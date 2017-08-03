import argparse
import logging
import os
import sys

from ..core.corpus.corpus import CorpusMetadata, StreamedCorpus, BatchedCorpus
from ..core.corpus.distribution import FreqDist

logger = logging.getLogger(__name__)


def main(corpus_path, output_dir, freq_dist_path=None):

    wordlist_meta = CorpusMetadata(
        name="Brysbaert 1 word",
        path="/Users/caiwingfield/code/corpus_analysis/scripts_corpus_info/brysbaert1.wordlist")

    token_delimiter = "\n"

    logger.info(f"Loading wordlist from {wordlist_meta.path}")
    with open(wordlist_meta.path, mode="r", encoding="utf-8") as wordlist_file:
        vocab_wordlist = set(wordlist_file.read().split(token_delimiter))
        logger.info(f"Wordlist has a vocab of size {len(vocab_wordlist):,}")

    logger.info(f"Loading corpus documents from {corpus_path}")
    vocab_corpus = set()
    token_i = 0
    for token in StreamedCorpus(CorpusMetadata(path=corpus_path, name="")):
        vocab_corpus.add(token)

        token_i += 1
        if token_i % 10_000_000 == 0:
            logger.info(f"\tRead {token_i:,} tokens")

    # The filter freqs we'll use
    filter_freqs = [0]
    if freq_dist_path is not None and os.path.isfile(freq_dist_path):
        logger.info("Loading frequency distribution")
        filter_freqs.extend([1, 5, 10, 25, 50, 100])
        freq_dist = FreqDist.load(freq_dist_path)
    else:
        logger.info("Building frequency distribution")
        freq_dist = FreqDist.from_batched_corpus(
            BatchedCorpus(CorpusMetadata(path=corpus_path, name=""), batch_size=1_000_000))

    for filter_freq in filter_freqs:

        info_filename = f"Brysbaert 1-word vocab overlap (filter {filter_freq}).info"

        # remove filtered terms from vocab
        filtered_vocab = vocab_corpus
        if filter_freq > 0:
            for token, freq in freq_dist.most_common():
                if freq <= filter_freq:
                    filtered_vocab.discard(token)

        with open(os.path.join(output_dir, info_filename), mode="w", encoding="utf-8") as info_file:
            message = f"Corpus (filter {filter_freq}) has a vocab of size {len(filtered_vocab):,}"
            log_and_write(message, info_file)

            overlap_vocab = set.intersection(filtered_vocab, vocab_wordlist)

            message = f"Overlap (filter {filter_freq}) has a size of\t{len(overlap_vocab):,}"
            log_and_write(message, info_file)

            missing_vocab = vocab_wordlist - filtered_vocab

            message = f"Missing words (filter {filter_freq}): {len(missing_vocab):,}"
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
    parser.add_argument("--freqdist", default=None)
    args = vars(parser.parse_args())

    main(corpus_path=args["corpuspath"], output_dir=args["outdir"], freq_dist_path=args["freqdist"])

    logger.info("Done!")
