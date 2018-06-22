"""
===========================
Tests for core.model classes.
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

import unittest
import math

import numpy

from ..core.model.ngram import LogNgramModel
from ..core.utils.constants import Chirality
from ..core.utils.maths import DistanceType
from ..core.corpus.corpus import BatchedCorpus
from ..core.corpus.indexing import FreqDist
from ..core.model.count import UnsummedCoOccurrenceCountModel, CoOccurrenceCountModel, LogCoOccurrenceCountModel, \
    ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel

from .testing_materials.metadata import test_corpus_metadata


class TestUnsummedCoOccurrenceModel(unittest.TestCase):
    def test_unsummed_cooccurrence_r1_left_values(self):
        model = UnsummedCoOccurrenceCountModel(test_corpus_metadata,
                                               window_radius=1,
                                               freq_dist=FreqDist.from_batched_corpus(
                                                   BatchedCorpus(test_corpus_metadata, 3)),
                                               chirality=Chirality.left)
        model.train(force_retrain=True)

        self.assertTrue(numpy.array_equal(
            model.matrix.todense(),
            numpy.array([
                [1, 1, 2, 0],
                [2, 0, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 0]
            ])
        ))

    def test_unsummed_cooccurrence_r1_right_values(self):
        model = UnsummedCoOccurrenceCountModel(test_corpus_metadata,
                                               window_radius=1,
                                               freq_dist=FreqDist.from_batched_corpus(
                                                   BatchedCorpus(test_corpus_metadata, 3)),
                                               chirality=Chirality.right)
        model.train(force_retrain=True)

        self.assertTrue(numpy.array_equal(
            model.matrix.todense(),
            numpy.array([
                [1, 2, 0, 1],
                [1, 0, 1, 0],
                [2, 0, 0, 0],
                [0, 0, 1, 0]
            ])
        ))


class TestCoOccurrenceModel(unittest.TestCase):
    def test_cooccurrence_contains_abcd(self):
        model = CoOccurrenceCountModel(test_corpus_metadata,
                                       window_radius=1,
                                       freq_dist=FreqDist.from_batched_corpus(
                                           BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        self.assertTrue(model.contains_word("A"))
        self.assertTrue(model.contains_word("B"))
        self.assertTrue(model.contains_word("C"))
        self.assertTrue(model.contains_word("D"))

    def test_cooccurrence_does_not_contain_e(self):
        model = CoOccurrenceCountModel(test_corpus_metadata,
                                       window_radius=1,
                                       freq_dist=FreqDist.from_batched_corpus(
                                           BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        self.assertFalse(model.contains_word("E"))

    def test_cooccurrence_r1_values(self):
        model = CoOccurrenceCountModel(test_corpus_metadata,
                                       window_radius=1,
                                       freq_dist=FreqDist.from_batched_corpus(
                                           BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        self.assertTrue(numpy.array_equal(
            model.matrix.todense(),
            numpy.array([
                [2, 3, 2, 1],
                [3, 0, 1, 0],
                [2, 1, 0, 1],
                [1, 0, 1, 0]
            ])
        ))

    def test_cooccurrence_r2_values(self):
        model = CoOccurrenceCountModel(test_corpus_metadata,
                                       window_radius=2,
                                       freq_dist=FreqDist.from_batched_corpus(
                                           BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        self.assertTrue(numpy.array_equal(
            model.matrix.todense(),
            numpy.array([
                [2, 5, 4, 3],
                [5, 0, 3, 0],
                [4, 3, 0, 1],
                [3, 0, 1, 0]
            ])
        ))

    def test_cooccurrence_r9_values(self):
        model = CoOccurrenceCountModel(test_corpus_metadata,
                                       window_radius=9,
                                       freq_dist=FreqDist.from_batched_corpus(
                                           BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        self.assertTrue(numpy.array_equal(
            model.matrix.todense(),
            numpy.array([
                [12, 12, 8, 4],
                [12, 6, 6, 3],
                [8, 6, 2, 2],
                [4, 3, 2, 0]
            ])
        ))

    def test_cooccurrence_r9_distance(self):
        model = CoOccurrenceCountModel(test_corpus_metadata,
                                       window_radius=9,
                                       freq_dist=FreqDist.from_batched_corpus(
                                           BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        self.assertAlmostEqual(
            model.distance_between("A", "B", DistanceType.Euclidean),
            math.sqrt((12 - 12) ** 2 + (6 - 12) ** 2 + (6 - 8) ** 2 + (3 - 4) ** 2)
        )


class TestLogCoOccurrence(unittest.TestCase):
    def test_log_cooccurrece_r1_values(self):
        model = LogCoOccurrenceCountModel(test_corpus_metadata,
                                          window_radius=1,
                                          freq_dist=FreqDist.from_batched_corpus(
                                              BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        try:
            numpy.testing.assert_array_almost_equal(
                model.matrix.todense(),
                numpy.log10(
                    numpy.array([
                        [2, 3, 2, 1],
                        [3, 0, 1, 0],
                        [2, 1, 0, 1],
                        [1, 0, 1, 0]
                    ])
                    + numpy.ones((4, 4))
                )
            )
        except AssertionError:
            self.fail("AssertionError raised by numpy.testing.assert_array_almost_equal.")


class TestConditionalProbability(unittest.TestCase):
    def test_conditional_probability_r1_values(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))
        model = ConditionalProbabilityModel(test_corpus_metadata,
                                            window_radius=1,
                                            freq_dist=fd)
        model.train(force_retrain=True)

        try:
            numpy.testing.assert_array_almost_equal(
                model.matrix.todense(),
                numpy.array([
                    [0.25, 0.375, 0.25, 0.125],
                    [0.75, 0.0, 0.25, 0.0],
                    [0.5, 0.25, 0.0, 0.25],
                    [0.5, 0.0, 0.5, 0.0]
                ])
            )
        except AssertionError:
            self.fail("AssertionError raised by numpy.testing.assert_array_almost_equal.")


class TestProbabilityRatios(unittest.TestCase):
    def test_probability_ratios_r1_values(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))
        model = ProbabilityRatioModel(test_corpus_metadata,
                                      window_radius=1,
                                      freq_dist=fd)
        model.train(force_retrain=True)

        try:
            numpy.testing.assert_array_almost_equal(
                model.matrix.todense(),
                numpy.array([
                    [0.625, 1.875, 1.25, 1.25],
                    [1.875, 0.0, 1.25, 0.0],
                    [1.25, 1.25, 0.0, 2.5],
                    [1.25, 0.0, 2.5, 0.0]
                ])
            )
        except AssertionError:
            self.fail("AssertionError raised by numpy.testing.assert_array_almost_equal.")


class TestPPMI(unittest.TestCase):
    def test_ppmi_r1_values(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))
        model = PPMIModel(test_corpus_metadata,
                          window_radius=1,
                          freq_dist=fd)
        model.train(force_retrain=True)

        desired_matrix = numpy.log2(numpy.array([
            [0.625, 1.875, 1.25, 1.25],
            [1.875, 0.0, 1.25, 0.0],
            [1.25, 1.25, 0.0, 2.5],
            [1.25, 0.0, 2.5, 0.0]
        ]))
        desired_matrix[desired_matrix < 0] = 0

        try:
            numpy.testing.assert_array_almost_equal(
                model.matrix.todense(),
                desired_matrix
            )
        except AssertionError:
            self.fail("AssertionError raised by numpy.testing.assert_array_almost_equal.")


class TestNgram(unittest.TestCase):
    def test_ngram_a_b(self):
        model = LogNgramModel(test_corpus_metadata,
                              window_radius=1,
                              freq_dist=FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3)))
        model.train(force_retrain=True)

        self.assertAlmostEqual(
            model.association_between("A", "B"),
            numpy.log10(3 + 1)
        )


if __name__ == '__main__':
    unittest.main()
