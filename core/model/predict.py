import logging
import os

import gensim

from ..utils.maths import Distance
from ..corpus.corpus import CorpusMetadata, BatchedCorpus
from ..model.base import VectorSpaceModel

logger = logging.getLogger(__name__)


# TODO: got more baseclassing to do with CBOW vs Skip-gram
class PredictModel(VectorSpaceModel):
    def __init__(self, model_type: VectorSpaceModel.Type, corpus: CorpusMetadata, vector_save_path,
                 window_radius: int, embedding_size: int):
        super().__init__(
            corpus=corpus,
            vector_save_path=vector_save_path,
            window_radius=window_radius,
            model_type=model_type)
        self.embedding_size = embedding_size
        self._corpus = BatchedCorpus(corpus, batch_size=1_000)

        self._model: gensim.models.Word2Vec = None

    def train(self, force_retrain: bool = False):

        if force_retrain or not os.path.isfile(self.vector_save_path):

            logger.info(f"Training {self.type.name} model")

            self._model = gensim.models.Word2Vec(
                # This is called "sentences", but they all get concatenated, so it doesn't matter.
                sentences=self._corpus,
                sg=self.type.sg_val,
                size=self.embedding_size,
                window=self.window_radius,
                # Recommended value from Mandera et al. (2017).
                # Baroni et al. (2014) recommend either 5 or 10, but 10 tended to perform slightly better overall.
                negative=10,
                # Recommended value from Mandera et al. (2017) and Baroni et al. (2014).
                sample=1e-5,
                # If we do filtering of word frequency, we'll do it in the corpus.
                min_count=0,
                workers=4)

        else:
            self.load()

    def save(self):
        self._model.save(self.vector_save_path)

    def nearest_neighbours(self, word: str, distance_type: Distance.Type, n: int):
        if distance_type is Distance.Type.cosine:
            # gensim implements cosine anyway, so this is an easy shortcut
            return self._model.wv.most_similar(positive=word, topn=n)
        else:
            # Other distances aren't implemented natively
            target_word = word
            target_vector = self.vector_for_word(target_word)

            nearest_neighbours = []

            for candidate_word in self._model.raw_vocab:

                # Skip target word
                if candidate_word is target_word:
                    continue

                candidate_vector = self.vector_for_word(candidate_word)
                distance_to_target = Distance.d(candidate_vector, target_vector, distance_type)

                # Add it to the shortlist
                nearest_neighbours.append((candidate_word, distance_to_target))
                nearest_neighbours.sort(key=lambda w, d: d)

                # If the list is overfull, remove the lowest one
                if len(nearest_neighbours) > n:
                    nearest_neighbours = nearest_neighbours[:-1]

            return [w for w, d in nearest_neighbours]

    def vector_for_word(self, word: str):
        return self._model.wv.word_vec(word, use_norm=True)

    def load(self):
        logger.info(f"Loading pre-trained {self.type.name} model")
        self._model = gensim.models.Word2Vec.load(self.vector_save_path)
