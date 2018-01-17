"""
===========================
Ngram models.
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
from abc import ABCMeta

from ..corpus.corpus import CorpusMetadata
from ..corpus.indexing import TokenIndexDictionary
from ..model.base import VectorSemanticModel, DistributionalSemanticModel
from ..model.count import CountVectorModel, LogCoOccurrenceCountModel
from ..utils.exceptions import WordNotFoundError

logger = logging.getLogger(__name__)


class NgramModel(DistributionalSemanticModel, metaclass=ABCMeta):
    """
    A model where vectors are computed by counting contexts.
    Essentially a wrapper for an underlying count model.
    """
    def __init__(self,
                 model_type: DistributionalSemanticModel.ModelType,
                 underlying_count_model: CountVectorModel,
                 ):
        super().__init__(model_type=model_type, corpus_meta=underlying_count_model.corpus_meta)
        self.underlying_count_model = underlying_count_model

    @property
    def window_radius(self) -> int:
        return self.underlying_count_model.window_radius

    @property
    def is_trained(self) -> bool:
        return self.underlying_count_model.is_trained

    def train(self, force_retrain: bool = False, memory_map: bool = False):
        self.underlying_count_model.train(force_retrain, memory_map)

    def untrain(self):
        self.underlying_count_model.untrain()

    @property
    def name(self) -> str:
        return f"{self.model_type.name} ({self.corpus_meta.name}), r={self.window_radius}"

    @property
    def _model_filename(self):
        return self.underlying_count_model._model_filename

    @property
    def _model_ext(self) -> str:
        return self.underlying_count_model._model_ext

    def _retrain(self):
        self.underlying_count_model._retrain()

    def _save(self):
        self.underlying_count_model._save()

    def _load(self, memory_map: bool = False):
        self.underlying_count_model._load(memory_map)

    def contains_word(self, word: str) -> bool:
        return self.underlying_count_model.contains_word(word)

    def association_between(self, word_1, word_2) -> float:
        """
        Returns the association between the two specified words.
        Not guaranteed to be symmetric.
        :param word_1:
        :param word_2:
        :return:
        :raises: WordNotFoundError
        """
        try:
            word_1_vector = self.underlying_count_model.vector_for_word(word_1)
        except KeyError:
            raise WordNotFoundError(f"The word '{word_1}' was not found.")

        try:
            word_2_index = self.underlying_count_model.token_indices.token2id[word_2]
        except KeyError:
            raise WordNotFoundError(f"The word '{word_2}' was not found.")

        return word_1_vector[0, word_2_index]


class LogNgramModel(NgramModel):
    """
    A model where the distance between word w and u is not the distance between their log n-gram vectors, but is the
    u-entry in the v-vector (which equals the v-entry in the u-vector as the log co-occurrence matrix is symmetric.

    log [ n(c,t) + 1 ]

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSemanticModel.ModelType.log_ngram,
                         LogCoOccurrenceCountModel(corpus_meta, window_radius, token_indices))
