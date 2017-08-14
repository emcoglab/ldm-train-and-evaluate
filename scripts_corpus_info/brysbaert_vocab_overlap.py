"""
===========================
Checks the overlap between the Brysbaert 40k norm words and a corpus.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

import logging
import os
import sys

from ..preferences.preferences import Preferences
from ..core.corpus.corpus import CorpusMetadata
from ..core.corpus.distribution import FreqDist

logger = logging.getLogger(__name__)


def main():

    wordlist_meta = CorpusMetadata(
        name="Brysbaert 1 word",
        path="/Users/caiwingfield/code/corpus_analysis/scripts_corpus_info/brysbaert1.wordlist")

    wordlist_delimiter = "\n"

    logger.info(f"Loading wordlist from {wordlist_meta.path}")
    with open(wordlist_meta.path, mode="r", encoding="utf-8") as wordlist_file:
        vocab_wordlist = set([token.lower() for token in wordlist_file.read().split(wordlist_delimiter)])
        logger.info(f"Wordlist has a vocab of size {len(vocab_wordlist):,}")

    for corpus_meta in Preferences.source_corpus_metas:

        freq_dist = FreqDist.load(corpus_meta.freq_dist_path)

        info_dir = os.path.dirname(corpus_meta.freq_dist_path)

        vocab_corpus = set(freq_dist.keys())

        # The filter freqs we'll use
        filter_freqs = [0, 1, 5, 10, 50, 100]

        for filter_freq in filter_freqs:

            info_filename = f"Brysbaert 1-word vocab overlap (filter {filter_freq}).info"

            # remove filtered terms from vocab
            filtered_vocab = vocab_corpus
            if filter_freq > 0:
                for token, freq in freq_dist.most_common():
                    if freq <= filter_freq:
                        filtered_vocab.discard(token)

            with open(os.path.join(info_dir, info_filename), mode="w", encoding="utf-8") as info_file:
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

    main()

    logger.info("Done!")
