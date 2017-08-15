#!/usr/bin/env bash

cd /Users/caiwingfield/code/

python -m corpus_analysis.scripts_clean_BBC.1_srt_deformat
python -m corpus_analysis.scripts_clean_BBC.2_remove_nonspeech
python -m corpus_analysis.scripts_clean_BBC.3_replace_problematic_characters
