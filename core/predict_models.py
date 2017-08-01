import logging
import os
from enum import Enum

import gensim

from core.corpus import BatchedCorpus

logger = logging.getLogger(__name__)


class PredictModelType(Enum):
    cbow = 0
    skip_gram = 1


class PredictModel(object):

    def __init__(self, predict_model_type, corpus_path, weights_path):
        """
        :param predict_model_type:
        :param corpus_path:
        :param weights_path:
        """
        self.predict_model_type = predict_model_type
        self.corpus_path = corpus_path
        self.weights_path = weights_path

        # Switch on predict model type
        if predict_model_type is PredictModelType.skip_gram:
            self._model_subdir = 'skip-gram'
            self._model_name = 'Skip-gram'
            self._sg = 1
        elif predict_model_type is PredictModelType.cbow:
            self._model_subdir = 'cbow'
            self._model_name = 'CBOW'
            self._sg = 0
        else:
            raise ValueError()

    def build_and_run(self):

        if not os.path.isfile(self.weights_path):

            logger.info(f"Training {self._model_name} model")

            # TODO: what size to use?
            embedding_dims = 100
            # TODO: run at different window radii
            window_radius = 5
            ignorable_frequency = 1

            # TODO: does using disjoint "sentences" here lead to unpleasant edge effects?
            corpus = BatchedCorpus(filename=self.corpus_path, batch_size=10)

            model = gensim.models.Word2Vec(
                sentences=corpus,
                size=embedding_dims,
                window=window_radius,
                min_count=ignorable_frequency,
                sg=self._sg,
                workers=4)

            model.save(self.weights_path)

        else:
            logger.info(f"Loading pre-trained {self._model_name} model")
            model = gensim.models.Word2Vec.load(self.weights_path)

        logger.info(f"For corpus {self.corpus_path}:")
        logger.info(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=4))


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
