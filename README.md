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

Ancillary computations in `core`.


## Running scripts

To run a script, you'll need to go above this project directory, and run using the `-m` flag.  For example:
```commandline
python -m corpus_analysis.scripts_model_evaluation.1_synonym_tests
```
This is because of weird behaviour of Python which I don't understand.


## Dependencies

This project requires Python 3.6+.

As well as things which come as standard with `conda`, this project uses the following modules which might not:

- `numpy`
- `scipy`
- `pandas`
- `seaborn`
- `lxml`
- `xlrd`
- `srt`
- `nltk`
- `gensim`
- `json`

Any which don't come with `conda` can be installed with `pip` like:
```commandline
pip install -U [package name]
```
