"""
===========================
Tests for core.utils.indexing.
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

from ..core.corpus.corpus import BatchedCorpus
from ..core.corpus.indexing import FreqDist, TokenIndex

from .testing_materials.metadata import test_corpus_metadata


class TokenIndexingTests(unittest.TestCase):
    """Tests for core.corpus.indexing.FreqDist involving token indexing."""

    def test_tid_is_invertible_token(self):
        ti = TokenIndex.from_freqdist_ranks(FreqDist.from_batched_corpus(
            BatchedCorpus(test_corpus_metadata, 3)))
        for token in ti.tokens:
            self.assertEqual(
                ti.id2token[ti.token2id[token]],
                token
            )

    def test_tid_is_invertible_id(self):
        ti = TokenIndex.from_freqdist_ranks(FreqDist.from_batched_corpus(
            BatchedCorpus(test_corpus_metadata, 3)))
        for i in range(len(ti)):
            self.assertEqual(
                ti.token2id[ti.id2token[i]],
                i
            )


class FreqDistTests(unittest.TestCase):
    """Tests for core.corpus.indexing.FreqDist involving frequency distributions."""

    def test_corpus_contains_specific_items(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))

        self.assertTrue("A" in fd.elements())
        self.assertTrue("B" in fd.elements())
        self.assertTrue("C" in fd.elements())
        self.assertTrue("D" in fd.elements())

    def test_corpus_has_specific_item_counts(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))

        self.assertEqual(fd["A"], 4)
        self.assertEqual(fd["B"], 3)
        self.assertEqual(fd["C"], 1)
        self.assertEqual(fd["D"], 2)

    def test_corpus_has_10_items(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))

        self.assertEqual(fd.N(), 10)


if __name__ == '__main__':
    unittest.main()
