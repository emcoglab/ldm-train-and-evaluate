"""
===========================
Global preference classes.
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

from ..core.corpus.corpus import CorpusMetadata


class Preferences(object):
    """
    Global preferences for models.
    """

    # Paths for intermediate processing steps
    bbc_processing_metas = dict(
        raw=CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC-mini/0 Raw"),
        no_srt=CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC-mini/1 No srt formatting"),
        no_nonspeech=CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC-mini/2 No nonspeech"),
        replaced_symbols=CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC/3 Replaced symbols"),
        tokenised=CorpusMetadata(
            name="BBC",
            path="/Users/caiwingfield/corpora/BBC/4 Tokenised/BBC.corpus",
            freq_dist_path="/Users/caiwingfield/corpora/BBC/4.1 info/frequency_distribution_BBC",
            index_path="/Users/caiwingfield/vectors/indexes/BBC.index"))
    bnc_processing_metas = dict(
        raw=CorpusMetadata(
            name="BNC",
            path="/Users/caiwingfield/corpora/BNC/0 XML version/Texts"),
        detagged=CorpusMetadata(
            name="BNC",
            path="/Users/caiwingfield/corpora/BNC/1 Detagged"),
        tokenised=CorpusMetadata(
            name="BNC",
            path="/Users/caiwingfield/corpora/BNC/2 Tokenised/BNC.corpus",
            freq_dist_path="/Users/caiwingfield/corpora/BNC/2.1 info/frequency_distribution_BNC",
            index_path="/Users/caiwingfield/vectors/indexes/BNC.index"))
    ukwac_processing_metas = dict(
        raw=CorpusMetadata(
            name="UKWAC",
            path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/0 Raw untagged/cleaned_pre.pos.corpus"),
        no_urls=CorpusMetadata(
            name="UKWAC",
            path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/1 Text only/cleaned_pre.pos.corpus"),
        partitioned=CorpusMetadata(
            name="UKWAC",
            path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/2 Partitioned"),
        tokenised=CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus",
            freq_dist_path="/Users/caiwingfield/corpora/UKWAC/3.1 info/frequency_distribution_UKWAC",
            index_path="/Users/caiwingfield/vectors/indexes/UKWAC.index"))

    # The final locations of the processed corpora
    source_corpus_metas = [
        bbc_processing_metas["tokenised"],
        bnc_processing_metas["tokenised"],
        # ukwac_processing_metas["tokenised"]
    ]

    # Word lists
    brysbaert_1w = CorpusMetadata(
        name="Brysbaert 1 word",
        path="/Users/caiwingfield/code/corpus_analysis/scripts_corpus_info/brysbaert1.wordlist")

    # We will test models with windows of each of these radii
    window_radii = [1, 3, 5, 10]

    # For the predict models, we will test a number of different embedding sizes
    # These sizes taken from Mandera et al. (2017)
    predict_embedding_sizes = [50, 100, 200, 300, 500]

    # The base directory for the models to be saved
    model_dir = "/Users/caiwingfield/vectors/"

    # Synonym tests
    toefl_question_path = "/Users/caiwingfield/evaluation/TOEFL/toefl.qst"
    toefl_answer_path = "/Users/caiwingfield/evaluation/TOEFL/toefl.ans"
    esl_test_path = "/Users/caiwingfield/evaluation/ESL/esl.txt"
    mcq_test_path = "/Users/caiwingfield/evaluation/LBM vocab MCQ/newMCQ.txt"

    # Model evaluation results
    eval_dir = "/Users/caiwingfield/evaluation/"
