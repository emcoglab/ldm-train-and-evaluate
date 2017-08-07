from abc import ABCMeta, abstractmethod
from enum import Enum, auto

from ..utils.maths import Distance
from ..corpus.corpus import CorpusMetadata


class VectorSpaceModel(metaclass=ABCMeta):
    class Type(Enum):
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
        def slug(self):
            """
            A path-safe representation of the model type
            :return:
            """
            if self is VectorSpaceModel.Type.cbow:
                return "cbow"
            elif self is VectorSpaceModel.Type.skip_gram:
                return "skipgram"
            elif self is VectorSpaceModel.Type.ngram_unsummed:
                return "ngram_unsummed"
            elif self is VectorSpaceModel.Type.ngram:
                return "ngram"
            elif self is VectorSpaceModel.Type.log_ngram:
                return "log_ngram"
            elif self is VectorSpaceModel.Type.ngram_probability:
                return "ngram_probability"
            elif self is VectorSpaceModel.Type.token_probability:
                return "token_probability"
            elif self is VectorSpaceModel.Type.context_probability:
                return "context_probability"
            elif self is VectorSpaceModel.Type.conditional_probability:
                return "conditional_probability"
            elif self is VectorSpaceModel.Type.probability_ratios:
                return "probability_ratios"
            elif self is VectorSpaceModel.Type.pmi:
                return "pmi"
            elif self is VectorSpaceModel.Type.ppmi:
                return "ppmi"
            else:
                raise ValueError()

        @property
        def name(self):
            """
            THe name of the model type
            :return:
            """
            if self is VectorSpaceModel.Type.cbow:
                return "CBOW"
            elif self is VectorSpaceModel.Type.skip_gram:
                return "Skip-gram"
            elif self is VectorSpaceModel.Type.ngram_unsummed:
                return "n-gram (unsummed)"
            elif self is VectorSpaceModel.Type.ngram:
                return "n-gram (summed)"
            elif self is VectorSpaceModel.Type.log_ngram:
                return "log n-gram"
            elif self is VectorSpaceModel.Type.ngram_probability:
                return "n-gramp robability"
            elif self is VectorSpaceModel.Type.token_probability:
                return "Token probability"
            elif self is VectorSpaceModel.Type.context_probability:
                return "Context probability"
            elif self is VectorSpaceModel.Type.conditional_probability:
                return "Conditional probability"
            elif self is VectorSpaceModel.Type.probability_ratios:
                return "Probability ratios"
            elif self is VectorSpaceModel.Type.pmi:
                return "PMI"
            elif self is VectorSpaceModel.Type.ppmi:
                return "PPMI"
            else:
                raise ValueError()

        @property
        def sg_val(self):
            """0 or 1"""
            if self is VectorSpaceModel.Type.cbow:
                return 0
            elif self is VectorSpaceModel.Type.skip_gram:
                return 1
            else:
                raise ValueError()
            
    def __init__(self, corpus_metadata: CorpusMetadata, model_type: Type, vector_save_path: str,
                 window_radius: int):
        self.window_radius = window_radius
        self.type = model_type
        self.vector_save_path = vector_save_path
        self.corpus_metadata = corpus_metadata

        # When implementing this class, this must be set by train()
        self._model = None

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

    def distance_between(self, word_1, word_2, distance_type: Distance.Type):
        """
        Returns the distance between the two specified words
        :param word_1:
        :param word_2:
        :param distance_type:
        :return:
        """
        return Distance.d(self.vector_for_word(word_1), self.vector_for_word(word_2), distance_type)
