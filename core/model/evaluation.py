import re
import typing

from abc import ABCMeta, abstractmethod

from ..utils.indexing import LetterIndexing


class SynonymQuestion(object):
    def __init__(self, prompt: str, options: typing.List[str], correct_i: int):
        self.correct_i = correct_i
        self.options = options
        self.prompt = prompt

    def answer_is_correct(self, guess_i: int) -> bool:
        """
        Check whether an answer to the question is correct.
        :param guess_i: 0-indexed
        :return:
        """
        return guess_i == self.correct_i


class SynonymTest(object, metaclass=ABCMeta):
    def __init__(self):
        # Backs self.question_list
        self._question_list: typing.List[SynonymQuestion] = None

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
    def n_questions(self) -> int:
        """
        The number of questions in the test.
        """
        return len(self._question_list)

    @abstractmethod
    def _load(self) -> typing.List[SynonymQuestion]:
        raise NotImplementedError()


class ToeflTest(SynonymTest):
    def _load(self) -> typing.List[SynonymQuestion]:
        # TODO: change these paths!
        question_path = "/Users/cai/Box Sync/LANGBOOT Project/Corpus Analysis/Synonym tests/TOEFL_BrEng/toefl.qst"
        answer_path = "/Users/cai/Box Sync/LANGBOOT Project/Corpus Analysis/Synonym tests/TOEFL_BrEng/toefl.ans"

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
        questions: typing.List[SynonymQuestion] = []
        with open(question_path, mode="r", encoding="utf-8") as question_file:
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
                questions.append(SynonymQuestion(prompt_word, option_list, -1))

                # There's a blank line after each question
                question_file.readline()

        # Get answers
        answers: typing.List[int] = []
        with open(answer_path, mode="r", encoding="utf-8") as answer_file:
            for answer_line in answer_file:
                answer_line = answer_line.strip()
                # Skip empty lines
                if answer_line == "":
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
    def _load(self) -> typing.List[SynonymQuestion]:
        test_path = "/Users/cai/Box Sync/LANGBOOT Project/Corpus Analysis/Synonym tests/ESL/esl.txt"
        question_re = re.compile(r"^"
                                 r"(?P<prompt_word>[a-z\-]+)"
                                 r"\s+\|\s+"
                                 r"(?P<option_list>[a-z\-\s|])"
                                 r"\s*$")

        questions: typing.List[SynonymQuestion] = []
        with open(test_path, mode="r", encoding="utf-8") as test_file:
            for line in test_file:
                line = line.strip()

                # Skip empty lines
                if not line:
                    break
                # Skip comments
                if line.startswith("#"):
                    continue

                question_match = re.match(question_re, line)

                prompt = question_match.group("prompt_word")
                options = [option.strip() for option in question_match.group("option_list").split("|")]

                # The first one is always the correct one
                questions.append(SynonymQuestion(prompt, options, correct_i=0))

        return questions
