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
import math
import logging
import os

from abc import ABCMeta, abstractmethod
from copy import copy
from typing import List

from ..model.base import VectorSemanticModel
from ..model.predict import PredictVectorModel
from ..utils.indexing import LetterIndexing
from ..utils.maths import DistanceType
from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


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
    def answer_word(self) -> str:
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
                f"Answer: {self.answer_word}.\t"
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


class ReportCard(object):
    """
    Description of the results of a battery of tests.
    """

    class Entry(object):
        """
        Description of the results of a test.
        """

        def __init__(self,
                     test: SynonymTest,
                     model: VectorSemanticModel,
                     distance_type: DistanceType,
                     answer_paper: AnswerPaper,
                     # ugh
                     append_to_model_name: str = ""):
            self._answer_paper = answer_paper
            self._distance_type = distance_type
            self._model_type_name = model.model_type.name + append_to_model_name
            self._window_radius = model.window_radius
            self._corpus_name = model.corpus_meta.name
            self._embedding_size = model.embedding_size if isinstance(model, PredictVectorModel) else None
            self._test_name = test.name

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
                f"{self._answer_paper.score}%"
            ]

    # An ordered list of the headings for each piece of data on the report card.
    headings = [
        "Test name",
        "Model",
        "Embedding size",
        "Radius",
        "Distance",
        "Corpus",
        "Score"
    ]

    def __init__(self):
        # Backing for self.entries property
        self._entries: List[ReportCard.Entry] = []

    def __iadd__(self, other: 'ReportCard'):
        for entry in other.entries:
            self.add_entry(entry)

    def __add__(self, other: 'ReportCard'):
        new = ReportCard()
        for entry in self.entries:
            new.add_entry(entry)
        for entry in other.entries:
            new.add_entry(entry)
        return new

    @property
    def entries(self) -> List[Entry]:
        """
        The entries on this report card.
        """
        return self._entries

    def add_entry(self, entry: Entry):
        """
        Adds an entry to the report card.
        """
        self._entries.append(entry)

    def save_csv(self,
                 csv_filename: str,
                 separator: str = ",",
                 include_headers: bool = True,
                 append_existing: bool = False
                 ):
        """
        Writes records to a CSV, creating it if it doesn't exist.
        :param append_existing: If True, an existing file will be appended
        :param csv_filename:
        :param separator:
        :param include_headers:
        Whether or not to include a header row. If appending, headers will not be written again even if this is True.
        """
        # Validate filename
        if not csv_filename.endswith(".csv"):
            csv_filename += ".csv"

        file_already_exists = os.path.isfile(csv_filename)

        if file_already_exists and append_existing:
            file_mode = "a"
        else:
            file_mode = "w"

        with open(csv_filename, mode=file_mode, encoding="utf-8") as csv_file:

            # Write headings
            if include_headers and not file_already_exists:
                csv_file.write(separator.join(ReportCard.headings) + "\n")

            # Write entries
            for entry in self.entries:
                csv_file.write(separator.join(entry.fields) + "\n")


class SynonymTester(object):
    """
    Administers a synonym test against a model.
    """

    def __init__(self, test_battery: List[SynonymTest]):
        self.test_battery = test_battery

    def administer_tests(self,
                         model: VectorSemanticModel,
                         truncate_vectors_at_length: int = None
                         ) -> ReportCard:
        """
        Administers a battery of tests against a model
        :param model: Must be trained.
        :param truncate_vectors_at_length:
        :return:
        """

        assert model.is_trained

        report_card = ReportCard()

        for distance_type in DistanceType:

            for test in self.test_battery:
                answers = []
                for question in test.question_list:
                    prompt = question.prompt
                    options = question.options

                    # The current best guess
                    best_guess_i = -1
                    best_guess_d = math.inf
                    for option_i, option in enumerate(options):
                        try:
                            guess_d = model.distance_between(prompt, option, distance_type,
                                                             truncate_vectors_at_length)
                        except KeyError as er:
                            logger.warning(f"{er.args[0]} was not found in the corpus.")
                            # Make sure we don't pick this one
                            guess_d = math.inf

                        if guess_d < best_guess_d:
                            best_guess_i = option_i
                            best_guess_d = guess_d

                    answers.append(AnsweredQuestion(copy(question), best_guess_i))

                answer_paper = AnswerPaper(answers)

                # Save the transcripts for this test
                answer_paper.save_text_transcript(self._text_transcript_path(test, distance_type, model,
                                                                             truncate_vectors_at_length))

                append_to_model_name = "" if truncate_vectors_at_length is None else f" ({truncate_vectors_at_length})"
                report_card.add_entry(ReportCard.Entry(test, model, distance_type, answer_paper,
                                                       append_to_model_name=append_to_model_name))

        return report_card

    @staticmethod
    def _text_transcript_path(test: SynonymTest, distance_type: DistanceType, model: VectorSemanticModel,
                              truncate_vectors_at_length: int) -> str:
        """
        Where the text transcript would be saved for a particular test.
        """
        filename = ""
        filename += f"{test.name}"
        filename += f" - {model.name}"
        filename += f" - {distance_type.name}"
        # Only record truncation of vectors if we're doing it
        filename += f" - s={truncate_vectors_at_length}" if truncate_vectors_at_length is not None else ""
        filename += f".txt"
        # TODO: this path shouldn't really be defined here
        return os.path.join(Preferences.eval_dir, "synonyms", "transcripts", filename)

    def all_transcripts_exist_for(self, model: VectorSemanticModel, truncate_vectors_at_length: int = None) -> bool:
        """
        If every test transcript file exists for this model.
        """
        for distance_type in DistanceType:
            for test in self.test_battery:
                # If one file doesn't exist
                if not os.path.isfile(self._text_transcript_path(test, distance_type, model,
                                                                 truncate_vectors_at_length)):
                    # The not all of them do
                    return False
        return True
