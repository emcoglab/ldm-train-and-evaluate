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

import os

from ..core.corpus.corpus import CorpusMetadata


class Preferences(object):
    """
    Global preferences for models.
    """

    # Paths for intermediate processing steps
    bnc_processing_metas = dict(
        raw=CorpusMetadata(
            name="BNC",
            path="/Volumes/Data/corpora/BNC/0 XML version/Texts"),
        detagged=CorpusMetadata(
            name="BNC",
            path="/Volumes/Data/corpora/BNC/1 Detagged"),
        tokenised=CorpusMetadata(
            name="BNC",
            path="/Volumes/Data/corpora/BNC/2 Tokenised/BNC.corpus",
            freq_dist_path="/Volumes/Data/corpora/BNC/2.1 info/frequency_distribution_BNC",
            index_path="/Volumes/Data/vectors/indexes/BNC.index"))
    bnc_text_processing_metas = dict(
        raw=CorpusMetadata(
            name="BNC-text",
            path="/Volumes/Data/corpora/BNC-text/0 XML version"),
        detagged=CorpusMetadata(
            name="BNC-text",
            path="/Volumes/Data/corpora/BNC-text/1 Detagged"),
        tokenised=CorpusMetadata(
            name="BNC-text",
            path="/Volumes/Data/corpora/BNC-text/2 Tokenised/BNC-text.corpus",
            freq_dist_path="/Volumes/Data/corpora/BNC-text/2.1 info/frequency_distribution_BNC_text",
            index_path="/Volumes/Data/vectors/indexes/BNC-text.index"))
    bnc_speech_processing_metas = dict(
        raw=CorpusMetadata(
            name="BNC-speech",
            path="/Volumes/Data/corpora/BNC-speech/0 XML version"),
        detagged=CorpusMetadata(
            name="BNC-speech",
            path="/Volumes/Data/corpora/BNC-speech/1 Detagged"),
        tokenised=CorpusMetadata(
            name="BNC-speech",
            path="/Volumes/Data/corpora/BNC-speech/2 Tokenised/BNC-speech.corpus",
            freq_dist_path="/Volumes/Data/corpora/BNC-speech/2.1 info/frequency_distribution_BNC_speech",
            index_path="/Volumes/Data/vectors/indexes/BNC-speech.index"))
    bbc_processing_metas = dict(
        raw=CorpusMetadata(
            name="BBC",
            path="/Volumes/Data/corpora/BBC-mini/0 Raw"),
        no_srt=CorpusMetadata(
            name="BBC",
            path="/Volumes/Data/corpora/BBC-mini/1 No srt formatting"),
        no_nonspeech=CorpusMetadata(
            name="BBC",
            path="/Volumes/Data/corpora/BBC-mini/2 No nonspeech"),
        replaced_symbols=CorpusMetadata(
            name="BBC",
            path="/Volumes/Data/corpora/BBC/3 Replaced symbols"),
        tokenised=CorpusMetadata(
            name="BBC",
            path="/Volumes/Data/corpora/BBC/4 Tokenised/BBC.corpus",
            freq_dist_path="/Volumes/Data/corpora/BBC/4.1 info/frequency_distribution_BBC",
            index_path="/Volumes/Data/vectors/indexes/BBC.index"))
    ukwac_processing_metas = dict(
        raw=CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/0 Raw untagged/cleaned_pre.pos.corpus"),
        no_urls=CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/1 Text only/cleaned_pre.pos.corpus"),
        partitioned=CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/2 Partitioned"),
        tokenised=CorpusMetadata(
            name="UKWAC",
            path="/Users/caiwingfield/corpora/UKWAC/3 Tokenised/UKWAC.corpus",
            freq_dist_path="/Users/caiwingfield/corpora/UKWAC/3.1 info/frequency_distribution_UKWAC",
            index_path="/Volumes/Data/vectors/indexes/UKWAC.index"))
    toy_processing_metas = dict(
        tokenised=CorpusMetadata(
            name="Toy",
            path="/Volumes/Data/corpora/toy-corpus/toy.corpus",
            freq_dist_path="/Volumes/Data/corpora/toy-corpus/frequency_distribution_toy",
            index_path="/Volumes/Data/corpora/toy-corpus/toy.index"))

    # The final locations of the processed corpora
    source_corpus_metas = [
        bnc_processing_metas["tokenised"],
        bbc_processing_metas["tokenised"],
        ukwac_processing_metas["tokenised"]
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
    model_dir                = "/Volumes/Data/vectors/"

    # TESTS
    test_dir                 = "/Users/caiwingfield/evaluation/tests/"

    # Synonym tests
    toefl_question_path      = os.path.join(test_dir, "TOEFL/toefl.qst")
    toefl_answer_path        = os.path.join(test_dir, "TOEFL/toefl.ans")
    esl_test_path            = os.path.join(test_dir, "ESL/esl.txt")
    mcq_test_path            = os.path.join(test_dir, "LBM vocab MCQ/newMCQ.txt")

    # Word similarity judgement tests
    simlex_path              = os.path.join(test_dir, "SimLex-999/SimLex-999.txt")
    wordsim_similarity_path  = os.path.join(test_dir, "WordSim353/wordsim_similarity_goldstandard.txt")
    wordsim_relatedness_path = os.path.join(test_dir, "WordSim353/wordsim_relatedness_goldstandard.txt")
    men_path                 = os.path.join(test_dir, "MEN/MEN_dataset_natural_form_full")

    # Semantic priming data
    spp_path_xls             = os.path.join(test_dir, "SPP/Hutchinson et al. (2013) SPP.xls")  # Semantic priming data: source xls file
    spp_elexicon_csv         = os.path.join(test_dir, "SPP/elexicon/I148613.csv")  # Additional Elexicon predictors csv

    # TEST RESULTS
    results_dir              = "/Users/caiwingfield/evaluation/results/"

    synonym_results_dir      = os.path.join(results_dir, "synonyms/")
    similarity_results_dir   = os.path.join(results_dir, "similarity/")
    spp_results_dir          = os.path.join(results_dir, "SPP/")
    spp_path_pickle          = os.path.join(results_dir, "SPP/Hutchinson et al. (2013) SPP.pickle")  # Semantic priming data: pickled version for faster loading
