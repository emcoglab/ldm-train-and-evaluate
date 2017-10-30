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

import numpy

from ..core.utils.constants import Chirality
from ..core.corpus.corpus import BatchedCorpus
from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.model.count import UnsummedNgramCountModel, NgramCountModel, LogNgramModel, ConditionalProbabilityModel, \
    ProbabilityRatioModel, PPMIModel

from .testing_materials.metadata import test_corpus_metadata


class TestUnsummedNgramModel(unittest.TestCase):
    def test_unsummed_ngram_r1_left_values(self):
        model = UnsummedNgramCountModel(test_corpus_metadata,
                                        window_radius=1,
                                        token_indices=TokenIndexDictionary.from_freqdist(FreqDist.from_batched_corpus(
                                            BatchedCorpus(test_corpus_metadata, 3))),
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

    def test_unsummed_ngram_r1_right_values(self):
        model = UnsummedNgramCountModel(test_corpus_metadata,
                                        window_radius=1,
                                        token_indices=TokenIndexDictionary.from_freqdist(FreqDist.from_batched_corpus(
                                            BatchedCorpus(test_corpus_metadata, 3))),
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


class TestSummedNgramModel(unittest.TestCase):
    def test_summed_ngram_r2_values(self):
        model = NgramCountModel(test_corpus_metadata,
                                window_radius=2,
                                token_indices=TokenIndexDictionary.from_freqdist(FreqDist.from_batched_corpus(
                                    BatchedCorpus(test_corpus_metadata, 3))))
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


class TestLogNgram(unittest.TestCase):
    def test_log_ngram_r1_values(self):
        model = LogNgramModel(test_corpus_metadata,
                              window_radius=1,
                              token_indices=TokenIndexDictionary.from_freqdist(FreqDist.from_batched_corpus(
                                  BatchedCorpus(test_corpus_metadata, 3))))
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
        tid = TokenIndexDictionary.from_freqdist(fd)
        model = ConditionalProbabilityModel(test_corpus_metadata,
                                            window_radius=1,
                                            token_indices=tid,
                                            freq_dist=fd)
        model.train(force_retrain=True)

        try:
            numpy.testing.assert_array_almost_equal(
                model.matrix.todense(),
                numpy.array([
                    [0.25,  0.375,  0.25,   0.125],
                    [0.75,  0.0,    0.25,   0.0  ],
                    [0.5,   0.25,   0.0,    0.25 ],
                    [0.5,   0.0,    0.5,    0.0  ]
                ])
            )
        except AssertionError:
            self.fail("AssertionError raised by numpy.testing.assert_array_almost_equal.")


class TestProbabilityRatios(unittest.TestCase):
    def test_probability_ratios_r1_values(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))
        tid = TokenIndexDictionary.from_freqdist(fd)
        model = ProbabilityRatioModel(test_corpus_metadata,
                                      window_radius=1,
                                      token_indices=tid,
                                      freq_dist=fd)
        model.train(force_retrain=True)

        try:
            numpy.testing.assert_array_almost_equal(
                model.matrix.todense(),
                numpy.array([
                    [0.625, 1.875,  1.25,   1.25],
                    [1.875, 0.0,    1.25,   0.0 ],
                    [1.25,  1.25,   0.0,    2.5 ],
                    [1.25,  0.0,    2.5,    0.0 ]
                ])
            )
        except AssertionError:
            self.fail("AssertionError raised by numpy.testing.assert_array_almost_equal.")


class TestPPMI(unittest.TestCase):
    def test_ppmi_r1_values(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))
        tid = TokenIndexDictionary.from_freqdist(fd)
        model = PPMIModel(test_corpus_metadata,
                          window_radius=1,
                          token_indices=tid,
                          freq_dist=fd)
        model.train(force_retrain=True)

        desired_matrix = numpy.log2(numpy.array([
            [0.625, 1.875,  1.25,   1.25],
            [1.875, 0.0,    1.25,   0.0 ],
            [1.25,  1.25,   0.0,    2.5 ],
            [1.25,  0.0,    2.5,    0.0 ]
        ]))
        desired_matrix[desired_matrix < 0] = 0

        try:
            numpy.testing.assert_array_almost_equal(
                model.matrix.todense(),
                desired_matrix
            )
        except AssertionError:
            self.fail("AssertionError raised by numpy.testing.assert_array_almost_equal.")


if __name__ == '__main__':
    unittest.main()
