# README #

Scripts for:

- Cleaning and tokenising text corpora.
- Computing summary information about text corpora.
- Building semantic vector space models from text corpora.
- Querying semantic vector space models using various distance measures.
- Evaluating semantic vector space models using a number of standard benchmarking tests.


## Structure

Scripts to run to reproduce the analysis are found in `scripts_…` directories; critical ones are numbered in sequence. 
Non-numbered scripts are just for fun.


## Running scripts

This project requires Python 3.7+.

To run a script, you'll need to go above this project directory, and run using the `-m` flag.  For example:
```commandline
python -m corpus_analysis.scripts_model_evaluation.1_synonym_tests
```
This is because of weird behaviour of Python which I don't understand.

