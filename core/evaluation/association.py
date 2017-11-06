"""
===========================
Testing against word association data.
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
import re
from abc import ABCMeta, abstractmethod
from typing import List

import numpy
import scipy.stats

from core.model.predict import PredictVectorModel
from .results import EvaluationResults
from ..model.base import VectorSemanticModel
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import DistanceType, CorrelationType
from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class AssociationResults(EvaluationResults):
    def __init__(self):
        super().__init__(
            results_column_names=["Correlation type", "Correlation"],
            save_dir=Preferences.association_results_dir
        )


class WordAssociation(object):
    """
    A judgement of the similarity between two words.
    """

    def __init__(self, word_1: str, word_2: str, association_strength: float):
        self.word_1 = word_1
        self.word_2 = word_2
        self.association_strength = association_strength


class WordAssociationTest(metaclass=ABCMeta):
    """
    A test against word-association data.
    """

    def __init__(self):
        # Backs self.judgement_list
        self._association_list: List[WordAssociation] = None

    @property
    def association_list(self) -> List[WordAssociation]:
        """
        The list of associations.
        """
        # Lazy load
        if self._association_list is None:
            self._association_list = self._load()
        assert self._association_list is not None
        return self._association_list

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the test.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load(self) -> List[WordAssociation]:
        """
        Load from source.
        """
        raise NotImplementedError()


# Static class
class AssociationTester(object):
    """
    Administers a word-association test against a model.
    """

    @staticmethod
    def administer_test(
            test: WordAssociationTest,
            model: VectorSemanticModel,
            distance_type: DistanceType
            ) -> AssociationResults:
        """
        Administers a battery of tests against a model
        :param model: Must be trained.
        """

        assert model.is_trained

        results = AssociationResults()

        for correlation_type in CorrelationType:
            human_judgements: List[WordAssociation] = []
            model_judgements: List[WordAssociation] = []
            for human_judgement in test.association_list:
                try:
                    distance = model.distance_between(
                        human_judgement.word_1,
                        human_judgement.word_2,
                        distance_type)
                except WordNotFoundError as er:
                    # If we can't find one of the words in the corpus, just ignore it.
                    logger.warning(er.message)
                    continue

                # If both words were found in the model, add them to the test list
                human_judgements.append(human_judgement)
                model_judgements.append(WordAssociation(
                    human_judgement.word_1,
                    human_judgement.word_2,
                    distance))

            # Apply correlation
            if correlation_type is CorrelationType.Pearson:
                correlation = numpy.corrcoef(
                    [j.association_strength for j in human_judgements],
                    [j.association_strength for j in model_judgements])[0][1]
            elif correlation_type is CorrelationType.Spearman:
                # PyCharm erroneously detects input types for scipy.stats.spearmanr as int rather than ndarray
                # noinspection PyTypeChecker,PyUnresolvedReferences
                correlation = scipy.stats.spearmanr(
                    [j.association_strength for j in human_judgements],
                    [j.association_strength for j in model_judgements]).correlation
            else:
                raise ValueError(correlation_type)

            results.add_result(test.name, model, distance_type, {"Correlation type": correlation_type.name, "Correlation": correlation})

        return results


class SimlexSimilarity(WordAssociationTest):
    """
    Simlex-999 judgements.
    """

    @property
    def name(self) -> str:
        return "Simlex-999"

    def _load(self) -> List[WordAssociation]:

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

            associations = []
            for line in simlex_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    associations.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("simlex_999"))))

        return associations


class MenSimilarity(WordAssociationTest):
    """
    MEN similarity judgements.
    From: Bruni, E., Tran, NK., Baroni, M. "Multimodal Distributional Semantics". J. AI Research. 49:1--47 (2014).
    """

    @property
    def name(self) -> str:
        return "MEN"

    def _load(self) -> List[WordAssociation]:

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
                    judgements.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("association"))))

        return judgements


class WordsimSimilarity(WordAssociationTest):
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
                    judgements.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("similarity"))))

        return judgements


class WordsimRelatedness(WordAssociationTest):
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
                    judgements.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("relatedness"))))

        return judgements


class ColourAssociation(WordAssociationTest):
    """
    Sutton & Altarriba (2016) colour associations.
    """

    @property
    def name(self) -> str:
        return "Colour associations"

    def _load(self) -> List[WordAssociation]:
        with open(Preferences.colour_association_path, mode="r", encoding="utf-8") as colour_assoc_file:
            # Skip header line
            colour_assoc_file.readline()
            assocs = []
            for line in colour_assoc_file:
                parts = line.split(",")
                assocs.append(WordAssociation(
                    # word
                    parts[1],
                    # colour
                    parts[2],
                    # percentage of respondents
                    float(parts[4])))

        return assocs


class ThematicAssociation(WordAssociationTest):
    """
    Jouravlev & McRae (2015) thematic associations.
    """

    @property
    def name(self) -> str:
        return "Thematic associations"

    def _load(self):
        # TODO
        raise NotImplementedError()
