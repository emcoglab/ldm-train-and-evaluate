"""
===========================
Base classes for language models.
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
from abc import ABCMeta, abstractmethod
from enum import Enum, auto

from ..corpus.corpus import CorpusMetadata
from ..utils.indexing import TokenIndexDictionary
from ..utils.maths import DistanceType, distance

logger = logging.getLogger(__name__)


class LanguageModel(metaclass=ABCMeta):
    """
    A model of the language.
    """

    class MetaType(Enum):
        count = auto()
        predict = auto()

    class ModelType(Enum):
        """
        Representative of the type of a vector space model.
        """
        # Predict model
        cbow = auto()
        skip_gram = auto()

        # Count model
        ngram_unsummed = auto()
        ngram = auto()
        log_ngram = auto()
        ngram_probability = auto()
        token_probability = auto()
        context_probability = auto()
        conditional_probability = auto()
        probability_ratios = auto()
        pmi = auto()
        ppmi = auto()

        @property
        def metatype(self):
            """
            The metatype of this type.
            :return:
            """
            if self is VectorSpaceModel.ModelType.cbow:
                return VectorSpaceModel.MetaType.predict
            elif self is VectorSpaceModel.ModelType.skip_gram:
                return VectorSpaceModel.MetaType.predict
            elif self is VectorSpaceModel.ModelType.ngram_unsummed:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.ngram:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.log_ngram:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.ngram_probability:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.token_probability:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.context_probability:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.conditional_probability:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.probability_ratios:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.pmi:
                return VectorSpaceModel.MetaType.count
            elif self is VectorSpaceModel.ModelType.ppmi:
                return VectorSpaceModel.MetaType.count
            else:
                raise ValueError()

        @property
        def slug(self):
            """
            A path-safe representation of the model type
            :return:
            """
            if self is VectorSpaceModel.ModelType.cbow:
                return "cbow"
            elif self is VectorSpaceModel.ModelType.skip_gram:
                return "skipgram"
            elif self is VectorSpaceModel.ModelType.ngram_unsummed:
                return "ngram_unsummed"
            elif self is VectorSpaceModel.ModelType.ngram:
                return "ngram"
            elif self is VectorSpaceModel.ModelType.log_ngram:
                return "log_ngram"
            elif self is VectorSpaceModel.ModelType.ngram_probability:
                return "ngram_probability"
            elif self is VectorSpaceModel.ModelType.token_probability:
                return "token_probability"
            elif self is VectorSpaceModel.ModelType.context_probability:
                return "context_probability"
            elif self is VectorSpaceModel.ModelType.conditional_probability:
                return "conditional_probability"
            elif self is VectorSpaceModel.ModelType.probability_ratios:
                return "probability_ratios"
            elif self is VectorSpaceModel.ModelType.pmi:
                return "pmi"
            elif self is VectorSpaceModel.ModelType.ppmi:
                return "ppmi"
            else:
                raise ValueError()

        @property
        def name(self):
            """
            THe name of the model type
            :return:
            """
            if self is VectorSpaceModel.ModelType.cbow:
                return "CBOW"
            elif self is VectorSpaceModel.ModelType.skip_gram:
                return "Skip-gram"
            elif self is VectorSpaceModel.ModelType.ngram_unsummed:
                return "n-gram (unsummed)"
            elif self is VectorSpaceModel.ModelType.ngram:
                return "n-gram (summed)"
            elif self is VectorSpaceModel.ModelType.log_ngram:
                return "log n-gram"
            elif self is VectorSpaceModel.ModelType.ngram_probability:
                return "n-gramp robability"
            elif self is VectorSpaceModel.ModelType.token_probability:
                return "Token probability"
            elif self is VectorSpaceModel.ModelType.context_probability:
                return "Context probability"
            elif self is VectorSpaceModel.ModelType.conditional_probability:
                return "Conditional probability"
            elif self is VectorSpaceModel.ModelType.probability_ratios:
                return "Probability ratios"
            elif self is VectorSpaceModel.ModelType.pmi:
                return "PMI"
            elif self is VectorSpaceModel.ModelType.ppmi:
                return "PPMI"
            else:
                raise ValueError()

        @classmethod
        def predict_types(cls):
            """
            Lists the predict types
            :return:
            """
            return [t for t in VectorSpaceModel.ModelType if t.metatype is VectorSpaceModel.MetaType.predict]

        @classmethod
        def count_types(cls):
            """
            Lists the count types
            :return:
            """
            return [t for t in VectorSpaceModel.ModelType if t.metatype is VectorSpaceModel.MetaType.count]

    def __init__(self,
                 model_type: ModelType,
                 corpus_meta: CorpusMetadata,
                 save_dir: str):
        self.model_type = model_type
        self.corpus_meta = corpus_meta
        # We need to remember the root directory for all models, as well as the save directory for this model.
        # This allows us to instantiate and load other models from the correct root.
        self._root_dir = save_dir

        # Must be set in __init__ of subclasses
        self._model_filename = None

    @property
    def save_dir(self) -> str:
        return os.path.join(self._root_dir, self.model_type.slug)

    @property
    def _previously_saved(self) -> bool:
        """
        Whether or not a previously saved model exists on the drive.
        """
        return os.path.isfile(os.path.join(self.save_dir, self._model_filename))

    def train(self, force_retrain: bool = False, load_if_previously_saved: bool = True):
        """
        Trains the model from its corpus, and saves the resultant state to drive.
        Will load existing model instead if possible.
        :param force_retrain: Retrain the model, even if there is a pre-existing saved state.
        :param load_if_previously_saved: Load a saved state if it's there to be loaded
        """
        if force_retrain or not self._previously_saved:
            logger.info(f"Training {self.model_type.name} model.")
            self._retrain()
            logger.info(f"Saving {self.model_type.name} model to {self._model_filename}.")
            self._save()
        elif load_if_previously_saved:
            logger.info(f"Loading {self.model_type.name} model from {self._model_filename}.")
            self._load()
        else:
            logger.info(f"Skipping {self.model_type.name} model: File exists {self._model_filename}.")
            pass

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    @abstractmethod
    def _load(self):
        """
        Loads a model.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def _save(self):
        """
        Saves a model in its current state.
        """
        raise NotImplementedError()


class ScalarModel(LanguageModel):
    """
    A language model where each word is associated with a scalar value.
    """

    def __init__(self,
                 model_type: LanguageModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(model_type, corpus_meta, save_dir)
        self.token_indices = token_indices
        self.window_radius = window_radius

        self._model_filename = f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type.name}"

        # When implementing this class, this must be set by retrain()
        self._model = None

    @property
    def vector(self):
        return self._model

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    @abstractmethod
    def _save(self):
        raise NotImplementedError()

    @abstractmethod
    def _load(self):
        raise NotImplementedError()

    @abstractmethod
    def scalar_for_word(self, word: str):
        """
        Returns the scalar value for a word.
        :param word:
        :return:
        """
        raise NotImplementedError()


class VectorSpaceModel(LanguageModel):
    """
    A language model where each word is associated with a point in a vector space.
    """

    def __init__(self,
                 model_type: LanguageModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int):
        super().__init__(model_type, corpus_meta, save_dir)
        self.window_radius = window_radius

        self._model_filename = f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type.slug}"

        # When implementing this class, this must be set by train()
        self._model = None

    @abstractmethod
    def vector_for_word(self, word: str):
        """
        Returns the vector representation of a word.
        :param word:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def nearest_neighbours(self, word: str, distance_type: DistanceType, n: int):
        """
        Finds the nearest neighbours to a word.
        """
        raise NotImplementedError()

    def nearest_neighbour(self, word: str, distance_type: DistanceType):
        """
        Finds the nearest neighbour to a word.
        :param word:
        :param distance_type:
        :return:
        """
        return self.nearest_neighbours(word, distance_type, 1)[0]

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    @abstractmethod
    def _load(self):
        raise NotImplementedError()

    @abstractmethod
    def _save(self):
        raise NotImplementedError()

    def distance_between(self, word_1, word_2, distance_type: DistanceType):
        """
        Returns the distance between the two specified words
        :param word_1:
        :param word_2:
        :param distance_type:
        :return:
        """
        return distance(self.vector_for_word(word_1), self.vector_for_word(word_2), distance_type)
