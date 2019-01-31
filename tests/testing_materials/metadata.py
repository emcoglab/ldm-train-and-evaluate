"""
===========================
Metadata for the test corpus.
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

import os

from ...ldm.corpus.corpus import CorpusMetadata

test_corpus_base_dir = os.path.dirname(os.path.realpath(__file__))


test_corpus_metadata = CorpusMetadata(
    name="test_corpus",
    path=os.path.join(test_corpus_base_dir, "test.corpus")
)
