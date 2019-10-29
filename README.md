# Train and evaluate linguistic distributional models

Repository hosted and maintained on Github: [https://github.com/emcoglab/ldm-train-and-evaluate](https://github.com/emcoglab/ldm-train-and-evaluate)

Scripts for:

- Cleaning and tokenising text corpora.
- Computing summary information about text corpora.
- Training linguistic distributional models (LDMs) from text corpora.
- Querying LDMs using various distance measures.
- Evaluating LDMs using several benchmarking test datasets.


## Running scripts

This project requires Python 3.7+.

Start by installing the requirements:
```commandline
pip install -r requirements.txt
```
You may want to use a [virtual environment](https://docs.python.org/3/library/venv.html).

To run a script, you'll need to go above this project directory, and run using the `-m` flag.  For example:
```commandline
python -m corpus_analysis.scripts_model_evaluation.1_synonym_tests
```


## Setting config

To set up config, copy the file `ldm/preferences/default_config.yaml` to somewhere else accessible and name it something like `congif_override.yaml`.  Then add the following as the first non-comment line in the script you are running:

    from ldm.preferences.config import Config; Config(use_config_overrides_from_file="/path/to/config_override.yaml")
    
Inside `config_override.yaml`, set the paths to be relevant to your local setup.  Only values set in `config_override.yaml` with override the corresponding value set in `default_config.yaml`, so you don't need to set everything if it's not relevant.


## Structure

Scripts to run to reproduce the analysis are found in `scripts_…` directories; critical ones are numbered in sequence. 
Non-numbered scripts are just for fun.

To run the analysis from beginning to end, run the following scripts in the following order and have a lot of time on your hands.
  
- `scripts_clean_BNC/1_separate_speech_and_text_documents.py`
- `scripts_clean_BNC/2_detag.py`

- `scripts_clean_BBC/1_srt_deformat.py`
- `scripts_clean_BBC/2_remove_nonspeech.py`
- `scripts_clean_BBC/3_replace_problematic_characters.py`
  
- `scripts_clean_UKWAC/1_remove_urls.py`
- `scripts_clean_UKWAC/2_partition.py`

- `scripts_corpus_common/1_tokenise.py`
- `scripts_corpus_common/2_frequency_distributions.py`

- `scripts_models/1_raw_cooccurrence_counts.py`
- `scripts_models/2_summed_ngrams.py`
- `scripts_models/3_log_ngrams.py`
- `scripts_models/4_ngram_probability.py`
- `scripts_models/5_token_probability.py`
- `scripts_models/6_context_probability.py`
- `scripts_models/7_conditional_probability.py`
- `scripts_models/8_probability_ratio.py`
- `scripts_models/9_pmi.py`
- `scripts_models/10_ppmi.py`
- `scripts_models/11_skipgram.py`
- `scripts_models/12_cbow.py`

- `scripts_model_evaluation/1_synonym_tests.py`
- `scripts_model_evaluation/2_word_associations.py`
- `scripts_model_evaluation/3_semantic_priming.py`
- `scripts_model_evaluation/4_concreteness_norms.py`
