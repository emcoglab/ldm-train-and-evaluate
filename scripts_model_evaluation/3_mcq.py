"""
===========================
Evaluate using TOEFL test.
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
import math
import sys

from ..core.model.count import PPMIModel
from ..core.model.evaluation import McqTest
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences
from ..core.corpus.distribution import FreqDist

logger = logging.getLogger(__name__)


def main():
    # TODO: Should work for all Preferences.window_radii
    window_radius = 1

    # TODO: Should work for all corpora
    corpus_metadata = Preferences.source_corpus_metas[0]  # BBC

    # TODO: Should work for vectors from all model types
    model = PPMIModel(corpus_metadata,
                      Preferences.model_dir,
                      window_radius,
                      TokenIndexDictionary.load(corpus_metadata.index_path),
                      FreqDist.load(corpus_metadata.freq_dist_path))

    model.train()

    mcq_test = McqTest()

    distance_type = DistanceType.correlation
    grades = []
    for mcq_question in mcq_test.question_list:
        prompt = mcq_question.prompt
        options = mcq_question.options

        # The current best guess:
        best_guess_i = -1
        best_guess_d = math.inf
        for guess_i, option in enumerate(options):
            try:
                guess_d = model.distance_between(prompt, option, distance_type)
            except KeyError as er:
                missing_word = er.args[0]
                logger.warning(f"{missing_word} was not found in the corpus.")
                # Make sure we don't pick this one
                guess_d = math.inf

            if guess_d < best_guess_d:
                best_guess_i, best_guess_d = guess_i, guess_d

        grades.append(int(mcq_question.answer_is_correct(best_guess_i)))

    logger.info(f"Score = {100 * sum(grades) / len(grades)}%")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
