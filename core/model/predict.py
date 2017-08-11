"""
===========================
Predict models.
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

from abc import abstractmethod

import gensim

from ..utils.maths import distance, DistanceType
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
        self._negative_sampling = 10
        # Recommended value from Mandera et al. (2017) and Baroni et al. (2014).
        self._sub_sample = 1e-5

        # Parallel workers
        self._workers = 4

        self._corpus = BatchedCorpus(corpus_meta, batch_size=1_000)
        self._model: gensim.models.Word2Vec = None

    @property
    def _model_filename(self):
        # Include embedding size
        return f"{self.corpus_meta.name}_r={self.window_radius}_s={self.embedding_size}_{self.model_type.slug}"

    def _save(self):
        logger.info(f"Saving {self.corpus_meta.name} to {self._model_filename}")
        self._model.save(os.path.join(self.save_dir, self._model_filename))

    def _load(self):
        logger.info(f"Loading {self.corpus_meta.name} from {self._model_filename}")
        self._model = gensim.models.Word2Vec.load(os.path.join(self.save_dir, self._model_filename))

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    def nearest_neighbours(self, word: str, distance_type: DistanceType, n: int):
        if distance_type is DistanceType.cosine:
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
                distance_to_target = distance(candidate_vector, target_vector, distance_type)

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

    def _retrain(self):

        logger.info(f"Corpus: {self.corpus_meta.name}, "
                    f"r={self.window_radius}, "
                    f"s={self.embedding_size}")

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

    def _retrain(self):

        logger.info(f"Corpus: {self.corpus_meta.name}, "
                    f"r={self.window_radius}, "
                    f"s={self.embedding_size}")

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