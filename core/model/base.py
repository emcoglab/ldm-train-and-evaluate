import os

from abc import ABCMeta, abstractmethod
from enum import Enum, auto

from ..utils.maths import Distance
from ..corpus.corpus import CorpusMetadata


class VectorSpaceModel(metaclass=ABCMeta):
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
            
    def __init__(self, corpus_meta: CorpusMetadata, model_type: ModelType, save_dir: str,
                 window_radius: int):
        self.window_radius = window_radius
        self.model_type = model_type
        self.save_dir = os.path.join(save_dir, model_type.slug)
        self.corpus_meta = corpus_meta

        # When implementing this class, this must be set by train()
        self._matrix = None

    @abstractmethod
    def vector_for_word(self, word: str):
        """
        Returns the vector representation of a word
        :param word:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def nearest_neighbours(self, word: str, distance_type: Distance.Type, n: int):
        """
        Finds the nearest neighbours to a word.
        """
        raise NotImplementedError()

    def nearest_neighbour(self, word: str, distance_type: Distance.Type):
        """
        Finds the nearest neighbour to a word.
        :param word:
        :param distance_type:
        :return:
        """
        return self.nearest_neighbours(word, distance_type, 1)[0]

    @abstractmethod
    def train(self, force_retrain: bool = False):
        """
        Trains the model from its corpus, and saves the resulting vectors to drive.
        Will load existing vectors instead if possible.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        """
        Loads a pretrained model.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        """
        Saves a model in its current state.
        """
        raise NotImplementedError()

    def distance_between(self, word_1, word_2, distance_type: Distance.Type):
        """
        Returns the distance between the two specified words
        :param word_1:
        :param word_2:
        :param distance_type:
        :return:
        """
        return Distance.d(self.vector_for_word(word_1), self.vector_for_word(word_2), distance_type)
