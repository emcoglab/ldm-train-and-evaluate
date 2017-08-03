import logging
import os
from enum import Enum

import gensim

from core.corpus.corpus import BatchedCorpus, CorpusMetadata

logger = logging.getLogger(__name__)


class PredictModelType(Enum):
    cbow = 0
    skip_gram = 1


class PredictModel(object):

    def __init__(self, model_type: PredictModelType, corpus_metadata: CorpusMetadata, weights_path, window_radius: int):
        """
        :type corpus_metadata: CorpusMetadata
        :type window_radius: int
        :param model_type: The type of model
        :param corpus_metadata: Where the corpus should be loaded from
        :param weights_path: Where the weights will be saved/loaded from
        """
        self.window_radius = window_radius
        self.model_type = model_type
        self.corpus = BatchedCorpus(corpus_metadata, batch_size=10)
        self.weights_path = weights_path

        # Switch on predict model type
        if model_type is PredictModelType.skip_gram:
            self._sg = 1
            self._model_name = 'Skip-gram'
        elif model_type is PredictModelType.cbow:
            self._sg = 0
            self._model_name = 'CBOW'
        else:
            raise ValueError()

        self.model: gensim.models.Word2Vec = None

    def build_and_run(self):

        if not os.path.isfile(self.weights_path):

            logger.info(f"Training {self._model_name} model")

            # TODO: what size to use?
            embedding_dims = 100
            # TODO: do we want to actually ignore low-frequency words?
            ignorable_frequency = 1

            self.model = gensim.models.Word2Vec(
                sentences=self.corpus,
                size=embedding_dims,
                window=self.window_radius,
                min_count=ignorable_frequency,
                sg=self._sg,
                workers=4)

            self.model.save(self.weights_path)

        else:
            logger.info(f"Loading pre-trained {self._model_name} model")
            self.model = gensim.models.Word2Vec.load(self.weights_path)


class PredictModelCBOW(PredictModel):
    def __init__(self, corpus_metadata, weights_path, window_radius):
        """
        :param corpus_metadata:
        :param weights_path:
        """
        super().__init__(PredictModelType.cbow, corpus_metadata, weights_path, window_radius)


class PredictModelSkipGram(PredictModel):
    def __init__(self, corpus_metadata, weights_path, window_radius):
        """
        :param corpus_metadata:
        :param weights_path:
        """
        super().__init__(PredictModelType.skip_gram, corpus_metadata, weights_path, window_radius)
