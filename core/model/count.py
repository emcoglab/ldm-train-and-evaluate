import logging
import os

import scipy.io as sio
import scipy.sparse as sps

from ..utils.maths import Distance
from ..corpus.corpus import CorpusMetadata, WindowedCorpus
from ..model.base import VectorSpaceModel
from ..utils.constants import Chiralities
from ..utils.indexing import TokenIndexDictionary

logger = logging.getLogger(__name__)


# TODO: got more baseclassing to do with all the different count model variants
class CountModel(VectorSpaceModel):
    def __init__(self, model_type: VectorSpaceModel.Type, corpus_metadata: CorpusMetadata, vector_save_path,
                 window_radius: int, token_indices: TokenIndexDictionary):
        super().__init__(
            corpus_metadata=corpus_metadata,
            model_type=model_type,
            vector_save_path=vector_save_path,
            window_radius=window_radius)

        self._token_indices = token_indices

        self._model = None

    def train(self, force_retrain: bool = False):

        vocab_size = len(self._token_indices)
        self._model = dict()

        for chi in Chiralities:

            cooccur_filename = os.path.join(
                self.vector_save_path,
                f"{self.corpus_metadata.name}_r={self.window_radius}_{chi}.cooccur")

            # Skip ones which are already done
            if os.path.isfile(cooccur_filename):
                logger.info(f"Loading {self.corpus_metadata.name} corpus, radius {self.window_radius}")
                self._model[chi] = sio.mmread(cooccur_filename)
            else:
                logger.info(f"Working on {self.corpus_metadata.name} corpus, radius {self.window_radius}")

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
                for window in WindowedCorpus(self.corpus_metadata, self.window_radius):

                    # The target token is the one in the middle, whose index is the radius of the window
                    target_token = window[self.window_radius]
                    target_id = self._token_indices.token2id[target_token]

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
                    context_id = self._token_indices.token2id[context_token]
                    cooccur[target_id, context_id] += 1

                    window_count += 1
                    if window_count % 1_000_000 == 0:
                        logger.info(f"\t{window_count:,} tokens processed")

                self._model[chi] = cooccur

                logger.info(f"Saving {chi}-cooccurrence matrix")
                sio.mmwrite(cooccur_filename, cooccur)

    def load(self):
        self._model = dict()
        for chi in Chiralities:
            logger.info(f"Loading {self.corpus_metadata.name} corpus, radius {self.window_radius}")
            cooccur_filename = os.path.join(
                self.vector_save_path,
                f"{self.corpus_metadata.name}_r={self.window_radius}_{chi}.cooccur")
            self._model[chi] = sio.mmread(cooccur_filename)

    def vector_for_id(self, word_id: int):
        """
        Returns the vector representation of a word, given by its index in the corpus.
        :param word_id:
        :return:
        """
        return self._model(word_id)

    def vector_for_word(self, word: str):
        word_id = self._token_indices.token2id(word)
        return self.vector_for_id(word_id)

    def nearest_neighbours(self, word: str, distance_type: Distance.Type, n: int):

        vocab_size = len(self._token_indices)

        target_id = self._token_indices.token2id(word)
        target_vector = self.vector_for_id(target_id)

        nearest_neighbours = []

        for candidate_id in range(0, vocab_size):

            # Skip target word
            if candidate_id == target_id:
                continue

            candidate_vector = self.vector_for_id(candidate_id)
            distance_to_target = Distance.d(candidate_vector, target_vector, distance_type)

            # Add it to the shortlist
            nearest_neighbours.append((candidate_id, distance_to_target))
            nearest_neighbours.sort(key=lambda c_id, c_dist: c_dist)

            # If the list is overfull, remove the lowest one
            if len(nearest_neighbours) > n:
                nearest_neighbours = nearest_neighbours[:-1]

        return [self._token_indices.id2token(i) for i, dist in nearest_neighbours]
