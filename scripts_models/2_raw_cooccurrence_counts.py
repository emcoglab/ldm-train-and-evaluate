import logging
import os
import sys

import scipy.io as sio
import scipy.sparse as sps

from .preferences import Preferences
from ..core.corpus.corpus import WindowedCorpus
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.constants import Chiralities

logger = logging.getLogger()


def main():

    out_dir = "/Users/caiwingfield/vectors/ngram"

    for meta in Preferences.source_corpus_metas:

        token_indices = TokenIndexDictionary.load(meta.index_path)
        vocab_size = len(token_indices)

        for radius in range(1, max(Preferences.window_radii) + 1):

            for chi in Chiralities:

                cooccur_filename = os.path.join(out_dir, f"{meta.name}_r={radius}_{chi}.cooccur")

                # Skip ones which are already done
                if os.path.isfile(cooccur_filename):
                    logger.info(f"Skipping {meta.name} corpus, radius {radius}")
                else:
                    logger.info(f"Working on {meta.name} corpus, radius {radius}")

                    # Initialise cooccurrence matrices

                    # We will store left- and right-cooccurrences separately.
                    # At this stage we define radius-n cooccurrence to be words which are /exactly/ n words apart,
                    # rather than /up-to/ n words apart.
                    # This will greatly speed up computation, and we can sum the values later much faster to get the
                    # standard "summed" n-gram counts.

                    # First coordinate points to target word
                    # Second coordinate points to context word
                    cooccur = sps.lil_matrix((vocab_size, vocab_size))

                    # Start scanning the corpus
                    window_count = 0
                    for window in WindowedCorpus(meta, radius):

                        # The target token is the one in the middle, whose index is the radius of the window
                        target_token = window[radius]
                        target_id = token_indices.token2id[target_token]

                        if chi is Chiralities.left:
                            # For the left occurrences, we look in the first position in the window; index 0
                            lr_index = 0
                        elif chi is Chiralities.right:
                            # For the right occurrences, we look in the last position in the window; index -1
                            lr_index = -1
                        else:
                            raise ValueError()

                        # Count lh occurrences
                        context_token = window[lr_index]
                        context_id = token_indices.token2id[context_token]
                        cooccur[target_id, context_id] += 1

                        window_count += 1
                        if window_count % 1_000_000 == 0:
                            logger.info(f"\t{window_count:,} tokens processed")

                    logger.info(f"Saving {chi}-cooccurrence matrix")
                    sio.mmwrite(cooccur_filename, cooccur)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
