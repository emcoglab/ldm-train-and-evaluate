#!/usr/bin/env bash

cd /Users/caiwingfield/code/

python -m corpus_analysis.scripts_models.1_raw_cooccurrence_counts
python -m corpus_analysis.scripts_models.2_summed_ngrams
python -m corpus_analysis.scripts_models.3_log_ngrams
python -m corpus_analysis.scripts_models.4_ngram_probability
python -m corpus_analysis.scripts_models.5_token_probability
python -m corpus_analysis.scripts_models.6_context_probability
python -m corpus_analysis.scripts_models.7_conditional_probability
python -m corpus_analysis.scripts_models.8_probability_ratio
python -m corpus_analysis.scripts_models.9_pmi
python -m corpus_analysis.scripts_models.10_ppmi
python -m corpus_analysis.scripts_models.11_skipgram
python -m corpus_analysis.scripts_models.12_cbow
