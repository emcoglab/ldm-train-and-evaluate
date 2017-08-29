"""
===========================
Testing against human similarity judgements.
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

import re
import logging

from abc import ABCMeta, abstractmethod
from typing import List

import numpy

from ..model.base import VectorSpaceModel
from ..model.predict import PredictModel
from ..utils.maths import DistanceType
from ...preferences.preferences import Preferences


logger = logging.getLogger(__name__)


class SimilarityJudgement(object):
    """
    A judgement of the similarity between two words
    """

    def __init__(self, word_1: str, word_2: str, similarity: float):
        self.word_1 = word_1
        self.word_2 = word_2
        self.similarity = similarity


class SimilarityJudgementTest(metaclass=ABCMeta):
    """
    A test against human similarity judgements of word pairs.
    """

    def __init__(self):
        # Backs self.judgement_list
        self._judgement_list: List[SimilarityJudgement] = None

    @property
    def judgement_list(self):
        """
        The list of judgements.
        """
        # Lazy load
        if self._judgement_list is None:
            self._judgement_list = self._load()
        return self._judgement_list

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the test.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load(self) -> List[SimilarityJudgement]:
        raise NotImplementedError()


class SimlexSimilarity(SimilarityJudgementTest):
    """
    Simlex-999 judgements.
    """

    @property
    def name(self) -> str:
        return "Simlex-999"

    def _load(self) -> List[SimilarityJudgement]:

        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              r"(?P<pos_tag>[A-Z])"  # The majority part-of-speech of the concept words, from BNC.
                              r"\s+"
                              r"(?P<simlex_999>[0-9.]+)"  # The SimLex999 similarity rating.  In range [0, 10].
                              r"\s+"
                              r"(?P<conc_w1>[0-9.]+)"  # Concreteness of word 1, from SFFAN.  In range [1, 7].
                              r"\s+"
                              r"(?P<conc_w2>[0-9.]+)"  # Concreteness of word 2, from SFFAN.  In range [1, 7].
                              r"\s+"
                              r"(?P<conc_q>[0-9])"  # The quartile the pair occupies.
                              r"\s+"
                              r"(?P<assoc_usf>[0-9.]+)"  # Strength of free association from word 1 to 2, from SFFAN.
                              r"\s+"
                              r"(?P<sim_assoc_333>[01])"  # Whether pair is among 333 most associated in the dataset. 
                              r"\s+"
                              r"(?P<sd_simlex>[0-9.]+)"  # The standard deviation of annotator scores.
                              r"\s*$")

        with open(Preferences.simlex_path, mode="r", encoding="utf-8") as simlex_file:
            # Skip header line
            simlex_file.readline()

            judgements = []
            for line in simlex_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(SimilarityJudgement(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("simlex_999"))))

        return judgements


class MenSimilarity(SimilarityJudgementTest):
    """
    MEN similarity judgements.
    From: Bruni, E., Tran, NK., Baroni, M. "Multimodal Distributional Semantics". J. AI Research. 49:1--47 (2014).
    """

    @property
    def name(self) -> str:
        return "MEN"

    def _load(self) -> List[SimilarityJudgement]:

        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s"
                              r"(?P<association>[0-9.]+)"  # Strength of association.
                              r"\s*$")

        with open(Preferences.men_path, mode="r", encoding="utf-8") as men_file:
            judgements = []
            for line in men_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(SimilarityJudgement(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("association"))))

        return judgements


class WordsimSimilarity(SimilarityJudgementTest):
    """
    WordSim-353 similarity judgements.
    """

    @property
    def name(self) -> str:
        return "WordSim-353 similarity"

    def _load(self):
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              # TODO: is it really in this range?
                              r"(?P<similarity>[0-9.]+)"  # The average similarity judgement.  In range [1, 10].
                              r"\s*$")

        with open(Preferences.wordsim_similarity_path, mode="r", encoding="utf-8") as wordsim_file:
            # Skip header line
            wordsim_file.readline()
            judgements = []
            for line in wordsim_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(SimilarityJudgement(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("similarity"))))

        return judgements


class WordsimRelatedness(SimilarityJudgementTest):
    """
    WordSim-353 relatedness judgements.
    """

    @property
    def name(self) -> str:
        return "WordSim-353 relatedness"

    def _load(self):
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              # TODO: is it really in this range? Does it matter?
                              r"(?P<relatedness>[0-9.]+)"  # The average relatedness judgement.  In range [1, 10].
                              r"\s*$")

        with open(Preferences.wordsim_relatedness_path, mode="r", encoding="utf-8") as wordsim_file:
            # Skip header line
            wordsim_file.readline()
            judgements = []
            for line in wordsim_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(SimilarityJudgement(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("relatedness"))))

        return judgements


class SimilarityTestResult(object):
    """
    Result of a similarity test.
    """

    def __init__(self,
                 model: VectorSpaceModel,
                 test: SimilarityJudgementTest,
                 distance_type: DistanceType,
                 correlation: float):
        self._test_name = test.name
        self._model_type_name = model.model_type.name
        self._window_radius = model.window_radius
        self._corpus_name = model.corpus_meta.name
        self._distance_type = distance_type
        self._embedding_size = model.embedding_size if isinstance(model, PredictModel) else None
        self._correlation = correlation

    @property
    def fields(self) -> List[str]:
        return [
            self._test_name,
            self._model_type_name,
            # Only PredictModels have an embedding size
            f"{self._embedding_size}" if self._embedding_size is not None else "",
            f"{self._window_radius}",
            self._distance_type.name,
            self._corpus_name,
            f"{self._correlation}"
        ]


class SimilarityTester(object):
    """
    Administers a synonym test against a model.
    """

    def __init__(self, test_battery: List[SimilarityJudgementTest]):
        self.test_battery = test_battery

    def administer_tests(self,
                         model: VectorSpaceModel) -> List[SimilarityTestResult]:
        """
        Administers a battery of tests against a model
        :param model: Must be trained.
        :return:
        """

        assert model.is_trained

        results: List[SimilarityTestResult] = []

        for distance_type in DistanceType:
            for test in self.test_battery:
                human_judgements: List[SimilarityJudgement] = []
                model_judgements: List[SimilarityJudgement] = []
                for human_judgement in test.judgement_list:
                    try:
                        distance = model.distance_between(
                            human_judgement.word_1,
                            human_judgement.word_2,
                            distance_type)
                    except KeyError as key_error:
                        # If we can't find one of the words in the corpus, just ignore it.
                        logger.warning(f"{model.corpus_meta.name} corpus doesn't contain {key_error.args[0]}")
                        continue

                    # If both words were found in the model, add them to the test list
                    human_judgements.append(human_judgement)
                    model_judgements.append(SimilarityJudgement(
                        human_judgement.word_1,
                        human_judgement.word_2,
                        distance))

                correlation = numpy.corrcoef(
                    [j.similarity for j in human_judgements],
                    [j.similarity for j in model_judgements])[0][1]

                results.append(SimilarityTestResult(model, test, distance_type, correlation))

        return results
