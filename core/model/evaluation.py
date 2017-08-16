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

import re
import typing
import math
import logging
import os

from abc import ABCMeta, abstractmethod
from copy import copy

from ..utils.maths import DistanceType
from ..model.base import VectorSpaceModel
from ...preferences.preferences import Preferences
from ..utils.indexing import LetterIndexing

logger = logging.getLogger(__name__)


class SynonymTestQuestion(object):
    """
    A synonym test question.
    """

    def __init__(self, prompt: str, options: typing.List[str], correct_i: int):
        self.correct_i = correct_i
        self.options = options
        self.prompt = prompt

    def __copy__(self):
        return SynonymTestQuestion(self.prompt, self.options.copy(), self.correct_i)


class AnsweredQuestion(object):
    """
    An answered synonym test question.
    """

    def __init__(self, question: SynonymTestQuestion, answer: int):
        self.answer = answer
        self.question = question

    @property
    def word(self) -> str:
        return self.question.options[self.answer]

    @property
    def is_correct(self):
        """
        Check whether an answer to the question is correct.
        """
        return self.question.correct_i == self.answer


class TestResults(object):
    """
    The results of a test
    """

    def __init__(self, transcript: typing.List[AnsweredQuestion]):
        self.transcript = transcript

    @property
    def _marks(self) -> typing.List[bool]:
        """
        List of correct/incorrect marks
        :return:
        """
        return [answer.is_correct for answer in self.transcript]

    @property
    def score(self) -> float:
        """
        The percentage of correct answers
        """
        return 100 * sum([int(m) for m in self._marks]) / len(self.transcript)


class SynonymTest(object, metaclass=ABCMeta):
    def __init__(self):
        # Backs self.question_list
        self._question_list: typing.List[SynonymTestQuestion] = None

    @property
    def question_list(self):
        """
        The list of questions.
        """
        # Lazy load
        if self._question_list is None:
            self._question_list = self._load()
        return self._question_list

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the test.
        """
        raise NotImplementedError()

    @property
    def n_questions(self) -> int:
        """
        The number of questions in the test.
        """
        return len(self._question_list)

    @abstractmethod
    def _load(self) -> typing.List[SynonymTestQuestion]:
        raise NotImplementedError()


class ToeflTest(SynonymTest):
    """
    TOEFL test.
    """

    @property
    def name(self):
        return "TOEFL"

    def _load(self) -> typing.List[SynonymTestQuestion]:

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
        questions: typing.List[SynonymTestQuestion] = []
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
        answers: typing.List[int] = []
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

    def _load(self) -> typing.List[SynonymTestQuestion]:
        question_re = re.compile(r"^"
                                 r"(?P<prompt_word>[a-z\-]+)"
                                 r"\s+\|\s+"
                                 r"(?P<option_list>[a-z\-\s|]+)"
                                 r"\s*$")

        questions: typing.List[SynonymTestQuestion] = []
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

    def _load(self) -> typing.List[SynonymTestQuestion]:

        n_options = 4
        questions: typing.List[SynonymTestQuestion] = []
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


class SynonymTester(object):
    def __init__(self,
                 model: VectorSpaceModel,
                 test: SynonymTest,
                 distance_type: DistanceType,
                 truncate_vectors_at_length: int = None
                 ):
        """
        Tests a model with a test.
        :param test: A synonym test.
        :param model: A model which must be TRAINED by the time the test is administered.
        :param distance_type:
        :param truncate_vectors_at_length:
        """

        self.distance_type = distance_type
        self.model = model
        self.test = test

        # Set by administer_test()
        self.results: TestResults = None

        self._truncate_at_length = truncate_vectors_at_length

    def administer_test(self):

        assert self.model.is_trained

        transcript = []
        for question in self.test.question_list:
            prompt = question.prompt
            options = question.options

            # The current best guess
            best_guess_i = -1
            best_guess_d = math.inf
            for option_i, option in enumerate(options):
                try:
                    guess_d = self.model.distance_between(prompt,
                                                          option,
                                                          self.distance_type,
                                                          self._truncate_at_length)
                except KeyError as er:
                    missing_word = er.args[0]
                    logger.warning(f"{missing_word} was not found in the corpus.")
                    # Make sure we don't pick this one
                    guess_d = math.inf

                if guess_d < best_guess_d:
                    best_guess_i = option_i
                    best_guess_d = guess_d

            answer = AnsweredQuestion(copy(question), best_guess_i)

            transcript.append(answer)

        self.results = TestResults(transcript)

    @property
    def _text_transcript_path(self) -> str:
        """
        Where the text transcript would be saved.
        """
        filename = ""
        filename += f"{self.test.name}"
        filename += f" - {self.model.corpus_meta.name}"
        filename += f" - {self.model.model_type.name}"
        filename += f" - r={self.model.window_radius}"
        filename += f" - {self.distance_type.name}"
        # Only record truncation of vectors if we're doing it
        filename += f" - s={self._truncate_at_length}" if self._truncate_at_length is not None else ""
        filename += f".txt"
        return os.path.join(Preferences.eval_dir, filename)

    @property
    def _evaluation_name(self) -> str:
        """
        Name for this evaluation.
        """
        name = ""
        name += f"{self.test.name} results for "
        name += f"{self.model.corpus_meta.name}, "
        name += f"{self.model.model_type.name}, "
        name += f"r={self.model.window_radius}, "
        name += f"{self.distance_type.name}"
        # Only record truncation of vectors if we're doing it
        name += f"s={self._truncate_at_length}" if self._truncate_at_length is not None else ""
        return name

    def save_text_transcript(self):
        """
        Saves a text transcript of the results of the test.
        """
        assert self.results is not None

        with open(self._text_transcript_path, mode="w", encoding="utf-8") as transcript_file:
            transcript_file.write(f"Transcript for {self._evaluation_name}\n")
            transcript_file.write("-----------------------\n")
            transcript_file.write(f"Overall score: {self.results.score}%\n")
            transcript_file.write("-----------------------\n")
            for line in self._get_text_transcript():
                transcript_file.write(line + "\n")

    @property
    def saved_transcript_exists(self):
        """
        Does a text transcript already exist?
        """
        return os.path.isfile(self._text_transcript_path)

    def _get_text_transcript(self) -> typing.List[str]:
        """
        Gets a text version of the results
        """
        assert self.results is not None
        page = []
        for answer in self.results.transcript:
            page.append(f"Q: {answer.question.prompt}?\t{' '.join(answer.question.options)}.\t"
                        f"A: {answer.word}:\t{'CORRECT' if answer.is_correct else 'INCORRECT'}")
        return page
