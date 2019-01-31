"""
===========================
Tests for core.evaluation.priming classes.
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

from ..ldm.evaluation.regression import SppData


class TestSppData(unittest.TestCase):
    def test_spp_prime_matched_word_spot_check(self):
        spp_data = SppData(force_reload=True, save_progress=False)

        for target_word in ["son", "quiet", "leaves", "frankenstein", "gas"]:

            matched_prime         = spp_data.dataframe.query(f"PrimeType == 'first_associate' & TargetWord == '{target_word}'")["MatchedPrime"].iloc[0]
            matched_matched_prime = spp_data.dataframe.query(f"PrimeType == 'first_unrelated' & TargetWord == '{target_word}'")["PrimeWord"].iloc[0]

            self.assertEqual(matched_prime, matched_matched_prime)
