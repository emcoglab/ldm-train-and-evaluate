"""
===========================
Tests for core.utils.maths.
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

from ..ldm.utils.maths import distance, DistanceType, levenshtein_distance, binomial_bayes_factor


class DistanceTests(unittest.TestCase):
    """Tests for core.utils.maths.distance function."""

    # Euclidean

    def test_euclidean_distance_from_vector_to_itself_is_0(self):
        u = numpy.array([1, 2, 3, 4])
        self.assertEqual(
            distance(u, u, DistanceType.Euclidean),
            0
        )

    def test_euclidean_distance_is_symmetric(self):
        u = numpy.array([2, 3, 4, 5])
        v = numpy.array([-7, -8, -9, 0])
        self.assertEqual(
            distance(u, v, DistanceType.Euclidean),
            distance(v, u, DistanceType.Euclidean)
        )

    # Cosine

    def test_cosine_distance_from_vector_to_itself_is_0(self):
        u = numpy.array([1, 2, 3, 4])
        self.assertEqual(
            distance(u, u, DistanceType.cosine),
            0
        )

    def test_cosine_distance_is_symmetric(self):
        u = numpy.array([2, 3, 4, 5])
        v = numpy.array([-7, -8, -9, 0])
        self.assertEqual(
            distance(u, v, DistanceType.cosine),
            distance(v, u, DistanceType.cosine)
        )

    # Correlation

    def test_correlation_distance_from_vector_to_itself_is_0(self):
        u = numpy.array([1, 2, 3, 4])
        self.assertEqual(
            distance(u, u, DistanceType.correlation),
            0)

    def test_correlation_distance_is_symmetric(self):
        u = numpy.array([2, 3, 4, 5])
        v = numpy.array([-7, -8, -9, 0])
        self.assertEqual(
            distance(u, v, DistanceType.correlation),
            distance(v, u, DistanceType.correlation)
        )


class LevenshteinDistanceTests(unittest.TestCase):
    """Tests for core.utils.maths.levenshtein_distance"""

    def test_old_insertion_cost_is_1(self):
        self.assertEqual(
            levenshtein_distance("a", "ab"),
            1
        )

    def test_old_deletion_cost_is_1(self):
        self.assertEqual(
            levenshtein_distance("ab", "a"),
            1
        )

    def test_old_substitution_cost_is_1(self):
        self.assertEqual(
            levenshtein_distance("ab", "ac"),
            1
        )

    def test_old_transposition_cost_is_2(self):
        self.assertEqual(
            levenshtein_distance("ab", "ba"),
            2
        )

    def test_old_from_string_to_itself_is_0(self):
        s = "a fancy string"
        self.assertEqual(
            levenshtein_distance(s, s),
            0
        )

    def test_old_is_symmetric(self):
        s = "a fancy string"
        t = "a more differenter string"
        self.assertEqual(
            levenshtein_distance(s, t),
            levenshtein_distance(t, s)
        )

    def test_specific_old(self):
        s = "quintessential"
        t = "quarantine"
        self.assertEqual(
            levenshtein_distance(s, t),
            9
        )


class BinomialBayesTests(unittest.TestCase):

    def test_one_sided_greater_binomial_bayes_10_7(self):
        # Value verified using Jasp
        self.assertAlmostEqual(
            binomial_bayes_factor(10, 7, 0.25, alternative_hypothesis=">", a=1, b=1),
            39.182,
            places=3
        )

    def test_one_sided_less_binomial_bayes_10_7(self):
        # Value verified using Jasp
        self.assertAlmostEqual(
            binomial_bayes_factor(10, 7, 0.25, alternative_hypothesis="<", a=1, b=1),
            0.140,
            places=3
        )

    def test_two_sided_binomial_bayes_10_7(self):
        # Value verified using Jasp
        self.assertAlmostEqual(
            binomial_bayes_factor(10, 7, 0.25, alternative_hypothesis="≠", a=1, b=1),
            29.421,
            places=3
        )


if __name__ == '__main__':
    unittest.main()
