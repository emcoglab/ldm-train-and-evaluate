import logging
import os
from enum import Enum

import gensim

from ..corpus.corpus import BatchedCorpus, CorpusMetadata

logger = logging.getLogger(__name__)


class PredictModelType(Enum):
    skip_gram = 1
    cbow = 0

    @property
    def slug(self):
        """
        A path-safe representation of the model type
        :return:
        """
        if self is PredictModelType.cbow:
            return "cbow"
        elif self is PredictModelType.skip_gram:
            return "skipgram"
        else:
            raise ValueError()

    @property
    def name(self):
        """
        THe name of the model type
        :return:
        """
        if self is PredictModelType.cbow:
            return "CBOW"
        elif self is PredictModelType.skip_gram:
            return "Skip-gram"
        else:
            raise ValueError()

    @property
    def sg_val(self):
        """0 or 1"""
        if self is PredictModelType.cbow:
            return 0
        elif self is PredictModelType.skip_gram:
            return 1
        else:
            raise ValueError()


class PredictModel(object):
    def __init__(self, model_type: PredictModelType, corpus_metadata: CorpusMetadata, weights_path, window_radius: int,
                 embedding_size: int):
        """
        :type corpus_metadata: CorpusMetadata
        :param model_type: The type of model
        :param corpus_metadata: Where the corpus should be loaded from
        :param weights_path: Where the weights will be saved/loaded from
        :type window_radius: int
        :type embedding_size: int
        """
        self.embedding_size = embedding_size
        self.window_radius = window_radius
        self.type = model_type
        self.corpus = BatchedCorpus(corpus_metadata,
                                    batch_size=1_000)
        self.weights_path = weights_path

        self.model: gensim.models.Word2Vec = None

    def build_and_run(self):

        if not os.path.isfile(self.weights_path):

            logger.info(f"Training {self.type.name} model")

            self.model = gensim.models.Word2Vec(
                # This is called "sentences", but they all get concatenated, so it doesn't matter.
                sentences=self.corpus,
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

            self.model.save(self.weights_path)

        else:
            logger.info(f"Loading pre-trained {self.type.name} model")
            self.model = gensim.models.Word2Vec.load(self.weights_path)
