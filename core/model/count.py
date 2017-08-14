"""
===========================
Count models.
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

from abc import abstractmethod, ABCMeta
from operator import itemgetter

import numpy as np
import scipy.io as sio
import scipy.sparse as sps

from ..corpus.corpus import CorpusMetadata, WindowedCorpus
from ..corpus.distribution import FreqDist
from ..model.base import VectorSpaceModel, LanguageModel
from ..utils.constants import Chirality
from ..utils.indexing import TokenIndexDictionary
from ..utils.maths import DistanceType, distance, sparse_max

logger = logging.getLogger(__name__)


class CountModel(VectorSpaceModel):
    """
    A model where vectors are computed by counting contexts.
    """

    def __init__(self,
                 model_type: LanguageModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(model_type, corpus_meta, save_dir, window_radius)
        self.token_indices = token_indices

    @property
    def _model_filename(self):
        # We're using Matrix Market format files
        return super()._model_filename + ".mtx"

    @property
    def matrix(self):
        return self._model

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    def _save(self):
        sio.mmwrite(os.path.join(self.save_dir, self._model_filename), self._model)

    def _load(self):
        self._model = sio.mmread(os.path.join(self.save_dir, self._model_filename)).tolil()

    # Overwrite to include .mtx extension
    @property
    def _previously_saved(self) -> bool:
        return os.path.isfile(os.path.join(self.save_dir, self._model_filename + ".mtx"))

    def vector_for_id(self, word_id: int):
        """
        Returns the vector representation of a word, given by its index in the corpus.
        :param word_id:
        :return:
        """
        # The first coordinate indexes target words, the second indexes context words.
        # So this should return a vector for the target word whose entries are indexed by context words.
        return self._model[word_id].todense()

    def vector_for_word(self, word: str):
        word_id = self.token_indices.token2id[word]
        return self.vector_for_id(word_id)

    def nearest_neighbours(self, word: str, distance_type: DistanceType, n: int):

        vocab_size = len(self.token_indices)

        target_id = self.token_indices.token2id[word]
        target_vector = self.vector_for_id(target_id)

        nearest_neighbours = []

        for candidate_id in range(0, vocab_size):

            # Skip target word
            if candidate_id == target_id:
                continue

            candidate_vector = self.vector_for_id(candidate_id)
            distance_to_target = distance(candidate_vector, target_vector, distance_type)

            # Add it to the shortlist
            nearest_neighbours.append((candidate_id, distance_to_target))

            # If the list is overfull, remove the lowest one
            if len(nearest_neighbours) > n:
                nearest_neighbours.sort(
                    # Sort by distance, which is the second item in the tuple.
                    key=itemgetter(1),
                    # Sort descending so the first element is the most similar
                    reverse=True)
                del nearest_neighbours[-1]

            if candidate_id % 10_000 == 0 and candidate_id > 0:
                logger.info(f'\t{candidate_id:,} out of {vocab_size:,} candidates considered. '
                            f'"{self.token_indices.id2token[nearest_neighbours[0][0]]}" currently the fave')

        return [self.token_indices.id2token(i) for i, dist in nearest_neighbours]


class ScalarCountModel(LanguageModel, metaclass=ABCMeta):
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
        self.window_radius = window_radius
        self.token_indices = token_indices

        # When implementing this class, this must be set by retrain()
        self._model: np.ndarray = None

    @property
    def name(self) -> str:
        return f"{self.model_type.name} ({self.corpus_meta.name}), r={self.window_radius}"

    # TODO: Rename this to put the type at the start!
    # @property
    def _model_filename(self):
        return f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type.name}.mtx"

    @property
    def vector(self):
        return self._model

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    def _save(self):
        sio.mmwrite(os.path.join(self.save_dir, self._model_filename), self._model)

    def _load(self):
        self._model = sio.mmread(os.path.join(self.save_dir, self._model_filename))

    def scalar_for_word(self, word: str):
        """
        Returns the scalar value for a word.
        :param word:
        :return:
        """
        return self._model[self.token_indices.token2id[word]]


class UnsummedNgramCountModel(CountModel):
    """
    A model where vectors consist of context counts at a fixed distance either on the left or right of a window.
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 chirality: Chirality):
        super().__init__(LanguageModel.ModelType.ngram_unsummed,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._chirality = chirality

    # Overwrite, to include chirality
    @property
    def name(self) -> str:
        return f"{self.model_type.name} ({self.corpus_meta.name}), r={self.window_radius}, {self._chirality.name}"

    # TODO: Rename this to put the type at the start!
    # Overwrite, to include chirality
    @property
    def _model_filename(self):
        return f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type.slug}_{self._chirality}.mtx"

    def _retrain(self):

        vocab_size = len(self.token_indices)

        # Initialise cooccurrence matrices

        # We will store left- and right-cooccurrences separately.
        # At this stage we define radius-n cooccurrence to be words which are /exactly/ n words apart,
        # rather than /up-to/ n words apart.
        # This will greatly speed up computation, and we can sum the values later much faster to get the
        # standard "summed" n-gram counts.

        # First coordinate points to target word
        # Second coordinate points to context word
        self._model = sps.lil_matrix((vocab_size, vocab_size))

        # Start scanning the corpus
        window_count = 0
        for window in WindowedCorpus(self.corpus_meta, self.window_radius):

            # The target token is the one in the middle, whose index is the radius of the window
            target_token = window[self.window_radius]
            target_id = self.token_indices.token2id[target_token]

            if self._chirality is Chirality.left:
                # For the left occurrences, we look in the first position in the window; index 0
                lr_index = 0
            elif self._chirality is Chirality.right:
                # For the right occurrences, we look in the last position in the window; index -1
                lr_index = -1
            else:
                raise ValueError()

            # Count lh occurrences
            context_token = window[lr_index]
            context_id = self.token_indices.token2id[context_token]
            self._model[target_id, context_id] += 1

            window_count += 1
            if window_count % 1_000_000 == 0:
                logger.info(f"\t{window_count:,} tokens processed")


class NgramCountModel(CountModel):
    """
    A model where vectors consist of the counts of context words within a window.

    n(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.ngram, corpus_meta, save_dir, window_radius, token_indices)

    def _retrain(self):

        vocab_size = len(self.token_indices)
        self._model = sps.lil_matrix((vocab_size, vocab_size))

        # We load the unsummed cooccurrence matrices in in sequence, and accumulate them to save the summed
        for radius in range(1, self.window_radius + 1):
            # Accumulate both left and right occurrences
            for chirality in Chirality:
                # Get each unsummed model
                unsummed_model = UnsummedNgramCountModel(self.corpus_meta, self._root_dir, radius, self.token_indices,
                                                         chirality)
                unsummed_model.train()

                # And add it to the current matrix
                self._model += unsummed_model.matrix

                # Prompt GC
                del unsummed_model


class LogNgramModel(CountModel):
    """
    A model where vectors consist of the log of context counts within a window.

    log n(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.log_ngram,
                         corpus_meta, save_dir, window_radius, token_indices)

    def _retrain(self):
        # Get the ngram model
        ngram_model = NgramCountModel(self.corpus_meta, self._root_dir, self.window_radius, self.token_indices)
        ngram_model.train()

        # Apply log to entries in the ngram matrix
        # Need to convert to csr first so that the log10 function will work
        self._model = ngram_model.matrix.tocsr()
        del ngram_model
        self._model.data = np.log10(self._model.data)
        self._model = self._model.tolil()


class NgramProbabilityModel(CountModel):
    """
    A model where vectors consist of the probability that a given context is found within a window around the target.

    p(c,t) = n(c,t) / NW

    c: context token
    t: target token
    N: size of corpus
    W: width of window
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 freq_dist: FreqDist):
        super().__init__(VectorSpaceModel.ModelType.ngram_probability,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._freq_dist = freq_dist

    def _retrain(self):
        # Get the ngram model
        ngram_model = NgramCountModel(self.corpus_meta, self._root_dir, self.window_radius, self.token_indices)
        ngram_model.train()

        # The probability is just the ngram count, divided by the width of the window and the size of the corpus
        self._model = ngram_model.matrix
        del ngram_model
        # The width of the window is twice the radius
        # We don't do 2r+1 because we only count the context words, not the target word
        self._model /= self.window_radius * 2
        self._model /= self._freq_dist.N()


class TokenProbabilityModel(ScalarCountModel):
    """
    A model where ~vectors~ consist of the probability that any token is the target.

    p(t) = Sum_c n(c,t) / NW

    c: context token
    t: target token
    N: size of corpus
    W: width of window
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 freq_dist: FreqDist):
        super().__init__(VectorSpaceModel.ModelType.token_probability,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._freq_dist = freq_dist

    def _retrain(self):
        # Get the ngram model
        ngram_model = NgramCountModel(self.corpus_meta, self._root_dir, self.window_radius, self.token_indices)
        ngram_model.train()

        # The probability is just the token count, divided by the width of the window and the size of the corpus
        # We're summing over contexts (second dim) to get a count of the targets
        self._model = np.sum(ngram_model.matrix, 1)
        del ngram_model
        # The width of the window is twice the radius
        # We don't do 2r+1 because we only count the context words, not the target word
        self._model /= self.window_radius * 2
        self._model /= self._freq_dist.N()


class ConditionalProbabilityModel(CountModel):
    """
    A model where vectors consist of n-gram counts normalised by token probabilities.

    p(c|t) = p(c,t) / p(t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 freq_dist: FreqDist):
        super().__init__(VectorSpaceModel.ModelType.conditional_probability,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._freq_dist = freq_dist

    def _retrain(self):
        ngram_probability_model = NgramProbabilityModel(self.corpus_meta, self._root_dir, self.window_radius,
                                                        self.token_indices, self._freq_dist)
        ngram_probability_model.train()

        # Convert to csr for linear algebra
        self._model = ngram_probability_model.matrix.tocsr()
        del ngram_probability_model

        token_probability_model = TokenProbabilityModel(self.corpus_meta, self._root_dir, self.window_radius,
                                                        self.token_indices, self._freq_dist)
        token_probability_model.train()

        # Here we divide each n-gram probability value by the token probability value.
        # This amounts to dividing each 0th-dim-slice of the matrix by a single value
        #
        #                                  p(c,t)         p(t)
        #
        #                               [ [-, -, -] ,     [ -,      <- entire mx row to be div'd by this vec entry
        # mx indexed by t on 0th dim ->   [-, -, -] ,  /    -,  <- vec indexed by t on 0th dim
        #                                 [-, -, -] ]       - ]
        #                                     ^
        #                                     |
        #                                     mx indexed by c on 1st dim
        #
        # According to https://stackoverflow.com/a/12238133/2883198, this is how you do that:
        self._model.data /= token_probability_model.vector.repeat(np.diff(self._model.indptr))
        self._model.tolil()


class ContextProbabilityModel(ScalarCountModel):
    """
    A model where ~vectors~ consist of the probability that any token is the target.

    p(c) = Sum_t n(c,t) / NW

    c: context token
    t: target token
    N: size of corpus
    W: width of window
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 freq_dist: FreqDist):
        super().__init__(VectorSpaceModel.ModelType.context_probability,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._freq_dist = freq_dist

    def _retrain(self):
        # Get the ngram model
        ngram_model = NgramCountModel(self.corpus_meta, self._root_dir, self.window_radius, self.token_indices)
        ngram_model.train()

        # The probability is just the token count, divided by the width of the window and the size of the corpus
        # We're summing over targets (first dim) to get the count of the contexts
        self._model = np.sum(ngram_model.matrix, 0)
        # The width of the window is twice the radius
        # We don't do 2r+1 because we only count the context words, not the target word
        self._model /= self.window_radius * 2
        self._model /= self._freq_dist.N()


class ProbabilityRatioModel(CountModel):
    """
    A model where vectors consist of the ratio of probabilities.

    r(c,t) = p(c|t) / p(c)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 freq_dist: FreqDist):
        super().__init__(LanguageModel.ModelType.probability_ratios,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._freq_dist = freq_dist

    def _retrain(self):
        ngram_model = NgramCountModel(self.corpus_meta, self._root_dir, self.window_radius, self.token_indices)
        ngram_model.train()

        # Convert to csr for linear algebra
        self._model = ngram_model.matrix.tocsr()
        del ngram_model

        context_probability_model = ContextProbabilityModel(self.corpus_meta, self._root_dir, self.window_radius,
                                                          self.token_indices, self._freq_dist)
        context_probability_model.train()

        # Here we divide each conditional n-gram probability value by the context probability value.
        # This amounts to dividing each 0th-dim-slice of the matrix by a single value
        #
        #                                  p(c|t)         p(c)
        #
        #                               [ [-, -, -] ,     [ -,      <- entire mx row to be div'd by this vec entry
        # mx indexed by t on 0th dim ->   [-, -, -] ,  /    -,  <- vec indexed by c on 0th dim
        #                                 [-, -, -] ]       - ]
        #                                     ^
        #                                     |
        #                                     mx indexed by c on 1st dim
        #
        # According to https://stackoverflow.com/a/12238133/2883198, this is how you do that:
        self._model.data /= context_probability_model.vector.repeat(np.diff(self._model.indptr))
        self._model.tolil()


class PMIModel(CountModel):
    """
    A model where the vectors consist of the pointwise mutual information between the context and the target.

    PMI(c,t) = log_2 r(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 freq_dist: FreqDist):
        super().__init__(LanguageModel.ModelType.pmi,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._freq_dist = freq_dist

    def _retrain(self):
        ratios_model = ProbabilityRatioModel(self.corpus_meta, self._root_dir, self.window_radius, self.token_indices,
                                             self._freq_dist)
        ratios_model.train()

        # Apply log to entries in the ngram matrix
        # Need to convert to csr first so that the log2 function will work
        self._model = ratios_model.matrix.tocsr()
        del ratios_model
        self._model.data = np.log2(self._model.data)
        self._model = self._model.tolil()


class PPMIModel(CountModel):
    """
    A model where the vectors consist of the positive pointwise mutual information between the context and the target.

    PMI^+(c,t) = max(0,PMI(c,t))

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 freq_dist: FreqDist):
        super().__init__(LanguageModel.ModelType.ppmi,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._freq_dist = freq_dist

    def _retrain(self):
        pmi_model = PMIModel(self.corpus_meta, self._root_dir, self.window_radius, self.token_indices, self._freq_dist)
        pmi_model.train()

        self._model = sparse_max(pmi_model.matrix,
                                 # same-shape zero matrix
                                 sps.lil_matrix(pmi_model.matrix.shape))
