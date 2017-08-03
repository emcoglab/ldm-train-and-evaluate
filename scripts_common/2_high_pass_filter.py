import logging
import os
import pickle
import sys

from ..core.corpus.corpus import CorpusMetadata, StreamedCorpus, BatchedCorpus
from ..core.corpus.distribution import FreqDistConstructor

logger = logging.getLogger(__name__)


def main():

    corpus_metas = [
        dict(
            source=CorpusMetadata(
                name="BBC",
                path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus",
                info_path="/Users/caiwingfield/corpora/BBC/4.1 info"),
            target=CorpusMetadata(
                name="BBC",
                path="/Users/caiwingfield/corpora/BBC/5 Filtered/BBC.corpus",
                info_path="/Users/caiwingfield/corpora/BBC/5.1 info")),
        dict(
            source=CorpusMetadata(
                name="BNC",
                path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus",
                info_path="/Users/caiwingfield/corpora/BNC/2.1 info"),
            target=CorpusMetadata(
                name="BNC",
                path="/Users/caiwingfield/corpora/BNC/3 Filtered/BNC.corpus",
                info_path="/Users/caiwingfield/corpora/BNC/3.1 info")),
        dict(
            source=CorpusMetadata(
                name="UKWAC",
                path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus",
                info_path="/Users/caiwingfield/corpora/UKWAC/3.1 info"),
            target=CorpusMetadata(
                name="UKWAC",
                path="/Users/caiwingfield/corpora/UKWAC/4 FIltered/UKWAC.corpus",
                info_path="/Users/caiwingfield/corpora/UKWAC/4.1 info"))]

    # The frequency at which we ignore tokens.
    # Set to 0 to include all tokens, set to 1 to include tokens that occur more than once, etc.
    ignorable_frequency = 1

    for corpus_meta in corpus_metas:

        freq_dist_path = os.path.join(corpus_meta["source"].info_path, corpus_meta["source"].name + ".corpus.pickle")
        if os.path.isfile(freq_dist_path):
            # If freq dist file previously saved, load it
            logger.info(f"Loading frequency distribution from {freq_dist_path}")
            with open(freq_dist_path, mode="rb") as freq_dist_file:
                freq_dist = pickle.load(freq_dist_file)
        else:
            # Compute it
            logger.info(f"Computing frequency distribution from {corpus_meta['source'].name} corpus")
            freq_dist = FreqDistConstructor.from_batched_corpus(
                BatchedCorpus(corpus_meta["source"], batch_size=1_000_000))

        logger.info(f"Loading {corpus_meta['source'].name} corpus from {corpus_meta['source'].path}")

        token_count = 0
        with open(corpus_meta["target"].path, mode="w", encoding="utf-8") as target_file:
            for token in StreamedCorpus(corpus_meta['source']):
                # Only write a token if it's sufficiently frequent
                if freq_dist[token] > ignorable_frequency:
                    target_file.write(token + "\n")

                    token_count += 1
                    if token_count % 1_000_000 == 0:
                        logging.info(f"\tWritten {token_count} tokens")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
