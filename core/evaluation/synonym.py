"""
===========================
Evaluation of models.
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
import math
import re

from abc import ABCMeta, abstractmethod
from copy import copy
from typing import List

from .results import ReportCard
from ..model.base import VectorSemanticModel
from ..utils.indexing import LetterIndexing
from ..utils.maths import DistanceType
from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class SynonymReportCard(ReportCard):
    def __init__(self):
        super().__init__()

    @classmethod
    def results_dir(cls) -> str:
        return Preferences.synonym_results_dir

    @classmethod
    def headings(cls) -> List[str]:
        return [
            "Test name",
            "Model",
            "Embedding size",
            "Radius",
            "Distance",
            "Corpus",
            "Score"
        ]

    class Entry(ReportCard.Entry):
        def __init__(self,
                     test: SynonymTest,
                     model: VectorSemanticModel,
                     distance_type: DistanceType,
                     answer_paper: AnswerPaper,
                     # ugh
                     append_to_model_name: str = ""):
            super().__init__(test.name, model.model_type.name + append_to_model_name, model, distance_type)
            self._answer_paper = answer_paper

        @property
        def fields(self):
            return [
                self._test_name,
                self._model_type_name,
                # Only PredictModels have an embedding size
                f"{self._embedding_size}" if self._embedding_size is not None else "",
                f"{self._window_radius}",
                self._distance_type.name,
                self._corpus_name,
                f"{self._answer_paper.score}%"
            ]


class SynonymTestQuestion(object):
    """
    A synonym test question.
    """

    def __init__(self, prompt: str, options: List[str], correct_i: int):
        self.correct_i = correct_i
        self.options = options
        self.prompt = prompt

    def __copy__(self):
        return SynonymTestQuestion(self.prompt, self.options.copy(), self.correct_i)

    @property
    def correct_answer_word(self) -> str:
        return self.options[self.correct_i]


class AnsweredQuestion(object):
    """
    An answered synonym test question.
    """

    def __init__(self, question: SynonymTestQuestion, answer_i: int):
        self.answer_i = answer_i
        self.question = question

    @property
    def correct_answer_word(self) -> str:
        return self.question.options[self.answer_i]

    @property
    def is_correct(self):
        """
        Check whether an answer to the question is correct.
        """
        return self.question.correct_i == self.answer_i

    def __str__(self):
        if self.is_correct:
            mark = "CORRECT"
        else:
            mark = f"INCORRECT ({self.question.correct_answer_word})"
        return (f"Question: {self.question.prompt}?\t"
                f"Options: {' '.join(self.question.options)}.\t"
                f"Answer: {self.correct_answer_word}.\t"
                f"Mark: {mark}")


class AnswerPaper(object):
    """
    The list of answered questions.
    """

    def __init__(self, answers: List[AnsweredQuestion]):
        self.answers = answers

    @property
    def score(self) -> float:
        """
        The percentage of correct answers.
        """
        return 100 * sum([int(answer.is_correct) for answer in self.answers]) / len(self.answers)

    def save_text_transcript(self, transcript_path: str):
        """
        Saves a text transcript of the results of the test.
        """
        # Validate filename
        if not transcript_path.endswith(".txt"):
            transcript_path += ".txt"

        with open(transcript_path, mode="w", encoding="utf-8") as transcript_file:
            transcript_file.write("-----------------------\n")
            transcript_file.write(f"Overall score: {self.score}%\n")
            transcript_file.write("-----------------------\n")
            for answer in self.answers:
                transcript_file.write(str(answer) + "\n")


class SynonymTest(object, metaclass=ABCMeta):
    def __init__(self):
        # Backs self.question_list
        self._question_list: List[SynonymTestQuestion] = None

    @property
    def question_list(self) -> List[SynonymTestQuestion]:
        """
        The list of questions.
        """
        # Lazy load
        if self._question_list is None:
            self._question_list = self._load()
        assert self._question_list is not None
        return self._question_list

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the test.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load(self) -> List[SynonymTestQuestion]:
        raise NotImplementedError()


class ToeflTest(SynonymTest):
    """
    TOEFL test.
    """

    @property
    def name(self) -> str:
        return "TOEFL"

    def _load(self) -> List[SynonymTestQuestion]:

        prompt_re = re.compile(r"^"
                               r"(?P<question_number>\d+)"
                               r"\.\s+"
                               r"(?P<prompt_word>[a-z\-]+)"
                               r"\s*$")
        option_re = re.compile(r"^"
                               r"(?P<option_letter>[a-d])"
                               r"\.\s+"
                               r"(?P<option_word>[a-z\-]+)"
                               r"\s*$")
        answer_re = re.compile(r"^"
                               r"(?P<question_number>\d+)"
                               # Who knows what
                               r"\s+\(a,a,-\)\s+\d+\s+"
                               r"(?P<option_letter>[a-d])"
                               r"\s*$")

        # Get questions
        n_options = 4
        questions: List[SynonymTestQuestion] = []
        with open(Preferences.toefl_question_path, mode="r", encoding="utf-8") as question_file:
            # Read groups of lines from file
            while True:
                prompt_line = question_file.readline().strip()

                # If we've reached the end of the file, stop reading
                if not prompt_line:
                    break

                prompt_match = re.match(prompt_re, prompt_line)
                prompt_word = prompt_match.group("prompt_word")

                option_list = []
                for option_i in range(n_options):
                    option_line = question_file.readline().strip()
                    option_match = re.match(option_re, option_line)
                    option_list.append(option_match.group("option_word"))

                # Using -1 as an "unset" value
                questions.append(SynonymTestQuestion(prompt_word, option_list, -1))

                # There's a blank line after each question
                question_file.readline()

        # Get answers
        answers: List[int] = []
        with open(Preferences.toefl_answer_path, mode="r", encoding="utf-8") as answer_file:
            for answer_line in answer_file:
                answer_line = answer_line.strip()
                # Skip empty lines
                if not answer_line:
                    continue

                answer_match = re.match(answer_re, answer_line)
                option_letter = answer_match.group("option_letter")
                answer_i = LetterIndexing.letter2int(option_letter)
                answers.append(answer_i)

        # Add the correct answers
        for question_i, question in enumerate(questions):
            question.correct_i = answers[question_i]

        return questions


class EslTest(SynonymTest):
    """
    ESL test.
    """

    @property
    def name(self):
        return "ESL"

    def _load(self) -> List[SynonymTestQuestion]:
        question_re = re.compile(r"^"
                                 r"(?P<prompt_word>[a-z\-]+)"
                                 r"\s+\|\s+"
                                 r"(?P<option_list>[a-z\-\s|]+)"
                                 r"\s*$")

        questions: List[SynonymTestQuestion] = []
        with open(Preferences.esl_test_path, mode="r", encoding="utf-8") as test_file:
            for line in test_file:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue
                # Skip comments
                if line.startswith("#"):
                    continue

                question_match = re.match(question_re, line)

                prompt = question_match.group("prompt_word")
                options = [option.strip() for option in question_match.group("option_list").split("|")]

                # The first one is always the correct one
                questions.append(SynonymTestQuestion(prompt, options, correct_i=0))

        return questions


class McqTest(SynonymTest):
    """
    MCQ test from Levy, Bullinaria and McCormick (2017).
    """

    @property
    def name(self):
        return "LBM's new MCQ"

    def _load(self) -> List[SynonymTestQuestion]:

        n_options = 4
        questions: List[SynonymTestQuestion] = []
        with open(Preferences.mcq_test_path, mode="r", encoding="utf-8") as test_file:
            while True:
                prompt = test_file.readline().strip()
                # Stop at the last line
                if not prompt:
                    break

                options = []
                for i in range(n_options):
                    options.append(test_file.readline().strip())

                # The first one is always the correct one
                questions.append(SynonymTestQuestion(prompt, options, correct_i=0))

        return questions


# Static class
class SynonymTester(object):
    """
    Administers a synonym test against a model.
    """

    @staticmethod
    def administer_tests(model: VectorSemanticModel,
                         test_battery: List[SynonymTest],
                         truncate_vectors_at_length: int = None
                         ) -> SynonymReportCard:
        """
        Administers a battery of tests against a model
        :param model: Must be trained.
        :param test_battery:
        :param truncate_vectors_at_length:
        :return:
        """

        assert model.is_trained

        report_card = SynonymReportCard()

        for distance_type in DistanceType:

            for test in test_battery:
                answers = []
                for question in test.question_list:
                    answer = SynonymTester.attempt_question(question, model, distance_type, truncate_vectors_at_length)

                    answers.append(answer)

                answer_paper = AnswerPaper(answers)

                append_to_model_name = "" if truncate_vectors_at_length is None else f" ({truncate_vectors_at_length})"
                report_card.add_entry(
                    SynonymReportCard.Entry(test, model, distance_type, answer_paper, append_to_model_name))

        return report_card

    @staticmethod
    def attempt_question(question: SynonymTestQuestion, model: VectorSemanticModel, distance_type: DistanceType,
                         truncate_vectors_at_length: int = None) -> AnsweredQuestion:
        """
        Attempt a question.
        :param question:
        :param model:
        :param distance_type:
        :param truncate_vectors_at_length:
        :return: answer
        """
        # The current best guess
        best_guess_i = -1
        best_guess_d = math.inf

        for option_i, option in enumerate(question.options):
            try:
                guess_d = model.distance_between(question.prompt, option, distance_type,
                                                 truncate_vectors_at_length)
            except KeyError as er:
                logger.warning(f"{er.args[0]} was not found in the corpus.")
                # Make sure we don't pick this one
                guess_d = math.inf

            if guess_d < best_guess_d:
                best_guess_i = option_i
                best_guess_d = guess_d

        return AnsweredQuestion(copy(question), best_guess_i)
