import logging
import os
from enum import Enum

import gensim

from core.corpus.corpus import BatchedCorpus

logger = logging.getLogger(__name__)


class PredictModelType(Enum):
    cbow = 0
    skip_gram = 1


class PredictModel(object):

    def __init__(self, model_type: PredictModelType, corpus_path, weights_path):
        """
        :param model_type: The type of model
        :param corpus_path: Where the corpus should be loaded from
        :param weights_path: Where the weights will be saved/loaded from
        """
        self.model_type = model_type
        self.corpus_path = corpus_path
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
            # TODO: run at different window radii
            window_radius = 5
            # TODO: do we want to actually ignore low-frequency words?
            ignorable_frequency = 1

            # TODO: does using disjoint "sentences" here lead to unpleasant edge effects?
            corpus = BatchedCorpus(filename=self.corpus_path, batch_size=10)

            self.model = gensim.models.Word2Vec(
                sentences=corpus,
                size=embedding_dims,
                window=window_radius,
                min_count=ignorable_frequency,
                sg=self._sg,
                workers=4)

            self.model.save(self.weights_path)

        else:
            logger.info(f"Loading pre-trained {self._model_name} model")
            self.model = gensim.models.Word2Vec.load(self.weights_path)


class PredictModelCBOW(PredictModel):
    def __init__(self, corpus_path, weights_path):
        """
        :param corpus_path:
        :param weights_path:
        """
        super().__init__(PredictModelType.cbow, corpus_path, weights_path)


class PredictModelSkipGram(PredictModel):
    def __init__(self, corpus_path, weights_path):
        """
        :param corpus_path:
        :param weights_path:
        """
        super().__init__(PredictModelType.skip_gram, corpus_path, weights_path)
