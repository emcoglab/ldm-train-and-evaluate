"""
===========================
Search for a word in a corpus.
Requires the corpus to have been processed with cleaning, common and info scripts.
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

import argparse

from ..core.corpus.distribution import FreqDist
from ..preferences.preferences import Preferences


def main(args):

    corpus_name = args.corpus_name.lower()
    word = args.word

    if corpus_name == "bnc":
        corpus_metadata = Preferences.bnc_processing_metas["tokenised"]
    elif corpus_name == "bnctext":
        corpus_metadata = Preferences.bnc_text_processing_metas["tokenised"]
    elif corpus_name == "bncspeech":
        corpus_metadata = Preferences.bnc_speech_processing_metas["tokenised"]
    elif corpus_name == "bbc":
        corpus_metadata = Preferences.bbc_processing_metas["tokenised"]
    elif corpus_name == "ukwac":
        corpus_metadata = Preferences.ukwac_processing_metas["tokenised"]
    else:
        raise ValueError(f"Corpus {corpus_name} doesn't exist.")

    freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

    try:
        occurrences = freq_dist[word]
    except KeyError:
        occurrences = 0

    s = "" if occurrences == 1 else "s"
    print(f"Word '{word}' occurs in corpus '{corpus_metadata.name}' {occurrences} time{s}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search for a word in a corpus.")

    parser.add_argument("corpus_name", type=str, help="The name of the corpus.")
    parser.add_argument("word", type=str, help="The word to search.")

    main(parser.parse_args())
