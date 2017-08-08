import logging
import os

from abc import abstractmethod

import gensim

from ..utils.maths import Distance
from ..corpus.corpus import CorpusMetadata, BatchedCorpus
from ..model.base import VectorSpaceModel, LanguageModel

logger = logging.getLogger(__name__)


class PredictModel(VectorSpaceModel):
    """
    A vector space model where words are predicted rather than counted.
    """
    def __init__(self,
                 model_type: LanguageModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 embedding_size: int):
        super().__init__(model_type, corpus_meta, save_dir, window_radius)
        self.embedding_size = embedding_size

        # Recommended value from Mandera et al. (2017).
        # Baroni et al. (2014) recommend either 5 or 10, but 10 tended to perform slightly better overall.
        self._negative_sampling = 10,
        # Recommended value from Mandera et al. (2017) and Baroni et al. (2014).
        self._sub_sample = 1e-5,

        # Parallel workers
        self._workers = 4

        self._corpus = BatchedCorpus(corpus_meta, batch_size=1_000)

        self._weights_filename = f"{corpus_meta.name}_r={self.window_radius}_s={embedding_size}_{model_type.slug}"

        self._model: gensim.models.Word2Vec = None

    def save(self):
        self._model.save(os.path.join(self.save_dir, self._weights_filename))

    def load(self):
        logger.info(f"Loading pre-trained {self.model_type.name} model")
        self._model = gensim.models.Word2Vec.load(os.path.join(self.save_dir, self._weights_filename))

    @abstractmethod
    def train(self, force_retrain: bool = False):
        raise NotImplementedError()

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


class CbowModel(PredictModel):
    """
    A vector space model trained using CBOW.
    """
    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 embedding_size: int):
        super().__init__(VectorSpaceModel.ModelType.cbow,
                         corpus_meta, save_dir, window_radius, embedding_size)

    def train(self, force_retrain: bool = False):

        if force_retrain or not os.path.isfile(self.save_dir):

            logger.info(f"Training {self.model_type.name} model")

            self._model = gensim.models.Word2Vec(
                # This is called "sentences", but they all get concatenated, so it doesn't matter.
                sentences=self._corpus,
                # This is CBOW, so don't use Skip-gram
                sg=0,
                size=self.embedding_size,
                window=self.window_radius,
                negative=self._negative_sampling,
                sample=self._sub_sample,
                # If we do filtering of word frequency, we'll do it in the corpus.
                min_count=0,
                workers=self._workers)

        else:
            self.load()


class SkipGramModel(PredictModel):
    """
    A vector space model trained using Skip-gram.
    """
    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 embedding_size: int):
        super().__init__(VectorSpaceModel.ModelType.skip_gram,
                         corpus_meta, save_dir, window_radius, embedding_size)

    def train(self, force_retrain: bool = False):

        if force_retrain or not os.path.isfile(self.save_dir):

            logger.info(f"Training {self.model_type.name} model")

            self._model = gensim.models.Word2Vec(
                # This is called "sentences", but they all get concatenated, so it doesn't matter.
                sentences=self._corpus,
                # This is Skip-gram, so make sure we use it!
                sg=1,
                size=self.embedding_size,
                window=self.window_radius,
                negative=self._negative_sampling,
                sample=self._sub_sample,
                # If we do filtering of word frequency, we'll do it in the corpus.
                min_count=0,
                workers=self._workers)

        else:
            self.load()
