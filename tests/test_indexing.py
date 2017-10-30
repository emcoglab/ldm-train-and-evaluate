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
from ..core.corpus.indexing import TokenIndexDictionary, FreqDist


from .test_corpus.metadata import test_corpus_metadata


class TokenIndexDictionaryTests(unittest.TestCase):
    """Tests for core.corpus.indexing.TokenIndexDictionary."""

    def test_tid_is_invertible_token(self):
        tid = TokenIndexDictionary.from_freqdist(
            FreqDist.from_batched_corpus(
                BatchedCorpus(test_corpus_metadata, 3)))
        for token in tid.tokens:
            self.assertEqual(
                tid.id2token[tid.token2id[token]],
                token
            )

    def test_tid_is_invertible_id(self):
        tid = TokenIndexDictionary.from_freqdist(
            FreqDist.from_batched_corpus(
                BatchedCorpus(test_corpus_metadata, 3)))
        for i in range(len(tid)):
            self.assertEqual(
                tid.token2id[tid.id2token[i]],
                i
            )


class FreqDistTests(unittest.TestCase):
    """Tests for core.corpus.indexing.FreqDist."""

    def test_specific_vaules_for_freqdist_from_batched_corpus(self):
        fd = FreqDist.from_batched_corpus(BatchedCorpus(test_corpus_metadata, 3))

        self.assertTrue("A" in fd.items())
        self.assertTrue("B" in fd.items())
        self.assertTrue("C" in fd.items())
        self.assertTrue("D" in fd.items())

        self.assertEqual(fd["A"], 4)
        self.assertEqual(fd["B"], 3)
        self.assertEqual(fd["C"], 1)
        self.assertEqual(fd["D"], 2)



if __name__ == '__main__':
    unittest.main()
