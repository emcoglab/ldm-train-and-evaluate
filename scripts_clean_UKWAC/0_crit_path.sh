#!/usr/bin/env bash

cd /Users/caiwingfield/code/

python -m corpus_analysis.scripts_clean_UKWAC.1_remove_urls
python -m corpus_analysis.scripts_clean_UKWAC.2_partition
