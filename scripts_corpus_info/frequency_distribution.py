import os
import sys
import logging
import argparse

import nltk

import matplotlib.pyplot as pplot

from ..core.tokenising import modified_word_tokenize
from ..core.filtering import filter_punctuation


logger = logging.getLogger()


def save_frequency_distribution_graph(freq_dist, filename, corpus_name="corpus", top_n=0):
    """
    Saves a frequency distribution graph.
    :param corpus_name:
    :param freq_dist:
    :param filename:
    :param top_n: Only plot the top n most-frequent tokens. Set to 0 to plot all.
    :return:
    """

    if top_n is 0:
        samples = [item for item, _ in freq_dist.most_common()]
    else:
        samples = [item for item, _ in freq_dist.most_common(top_n)]

    fig = pplot.figure(num=None, figsize=(30, 20), dpi=192, facecolor='w', edgecolor='k')

    ax = pplot.gca()

    freqs = [freq_dist[sample] for sample in samples]

    if top_n > 0:
        title = f"Frequencies of top {top_n} {corpus_name} tokens"
    else:
        title = f"Frequencies of all {corpus_name} tokens"
    x_label = "Token"
    y_label = "Frequency"

    pplot.plot(freqs)

    pplot.yscale("log")
    pplot.ylim(ymin=1)
    pplot.draw()

    pplot.title(title)
    pplot.ylabel(y_label)
    pplot.xlabel(x_label)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels(samples, rotation="vertical")

    pad = 0.05
    pplot.subplots_adjust(top=1 - pad, bottom=0.15, left=pad, right=1 - pad)

    pplot.margins(0, 0)

    if not filename.endswith(".png"):
        filename += ".png"

    fig.savefig(filename)

    pplot.close(fig)

    return freq_dist


def save_frequency_distribution_info(freq_dist, filename):
    """
    Saves information about a nltk.probability.FreqDist
    :param freq_dist:
    :param filename:
    :return:
    """
    most_common = freq_dist.most_common()

    if not filename.endswith(".txt"):
        filename += ".txt"

    with open(filename, mode="w", encoding="utf-8") as info_file:

        # Write basic info
        info_file.write(f"Total number of tokens: {len(freq_dist)}\n")
        info_file.write(f"Total number of unique tokens: {len(most_common)}'\n")
        info_file.write(f"\n")

        # Write frequencies
        info_file.write(f"Frequencies:\n")
        info_file.write(f"\n")
        for token, count in most_common:
            info_file.write(f"{token}\t{count}\n")


def main(corpus_name, corpus_dir, output_dir):

    logger.info(f"Working on {corpus_name} corpus")

    logger.info(f"Loading corpus documents from {corpus_dir}")
    corpus = nltk.corpus.PlaintextCorpusReader(corpus_dir, ".+\..+")

    logger.info(f"Tokenising corpus")
    corpus = [w.lower() for w in modified_word_tokenize(corpus.raw())]

    logger.info(f"Filtering corpus")
    corpus = filter_punctuation(corpus)

    logger.info(f"Saving frequency distribution information")
    freq_dist = nltk.probability.FreqDist(corpus)
    save_frequency_distribution_info(
        freq_dist,
        os.path.join(output_dir, f"Frequency distribution info {corpus_name}.txt"))
    save_frequency_distribution_graph(
        freq_dist,
        os.path.join(output_dir, f"Frequency distribution graph{corpus_name}.png"),
        corpus_name=corpus_name,
        top_n=200)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_name")
    parser.add_argument("corpus_dir")
    parser.add_argument("output_dir")
    args = vars(parser.parse_args())

    main(corpus_name=args["corpus_name"], corpus_dir=args["corpus_dir"], output_dir=args["output_dir"])

    logger.info("Done!")