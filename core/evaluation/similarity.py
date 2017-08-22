import re

from abc import ABCMeta, abstractmethod
from typing import List

import numpy

from ..model.base import VectorSpaceModel
from ..model.predict import PredictModel
from ..utils.maths import DistanceType
from ...preferences.preferences import Preferences


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
                              # TODO: is it really in this range?
                              r"(?P<relatedness>[0-9.]+)"  # The average relatedness judgement.  In range [1, 10].
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
                        float(entry_match.group("relatedness"))))

        return judgements


class SimilarityTester(object):
    """
    Administers a synonym test against a model.
    """

    def __init__(self, test_battery: List[SimilarityJudgementTest]):
        self.test_battery = test_battery

    def administer_tests(self,
                         model: VectorSpaceModel):
        """
        Administers a battery of tests against a model
        :param model: Must be trained.
        :return:
        """

        assert model.is_trained

        for distance_type in DistanceType:
            for test in self.test_battery:
                model_judgements: List[SimilarityJudgement] = []
                for human_judgement in test.judgement_list:
                    model_judgements.append(SimilarityJudgement(
                        human_judgement.word_1,
                        human_judgement.word_2,
                        model.distance_between(
                            human_judgement.word_1,
                            human_judgement.word_2,
                            distance_type
                        )))

                correlation = numpy.corrcoef(
                    [j.similarity for j in test.judgement_list],
                    [j.similarity for j in model_judgements]
                )


