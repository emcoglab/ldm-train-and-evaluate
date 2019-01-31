"""
===========================
Compute and save frequency distribution information from a corpus.
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

import logging
import os
import sys

from ..ldm.utils.logging import log_message, date_format
from ..ldm.corpus.indexing import FreqDist
from ..ldm.preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def save_frequency_distribution_graph(freq_dist, filename, corpus_name="corpus", top_n=0):
    """
    Saves a frequency distribution graph.
    :param corpus_name:
    :param freq_dist:
    :param filename:
    :param top_n: Only plot the top n most-frequent tokens. Set to 0 to plot all.
    :return:
    """

    import matplotlib
    matplotlib.use('TkAgg')  # To run on MacOS
    import matplotlib.pyplot as pplot

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
    Saves information about a FreqDist
    :param freq_dist:
    :param filename:
    :return:
    """
    most_common = freq_dist.most_common()

    if not filename.endswith(".txt"):
        filename += ".txt"

    with open(filename, mode="w", encoding="utf-8") as info_file:

        # Write basic info
        info_file.write(f"Vocabulary size: {len(most_common):,}\n")

        # Write low-frequency counts
        for cutoff_freq in [0, 1, 5, 10, 50, 100]:
            info_file.write(f"Tokens occurring ≥ {cutoff_freq} times."
                            f"\tCorpus size:"
                            f"\t{sum([count for token, count in most_common if count > cutoff_freq]):,}."
                            f"\tVocab size:"
                            f"\t{len([count for token, count in most_common if count > cutoff_freq]):,}."
                            f"\n")

        info_file.write("\n")
        info_file.write("----------------------------\n")
        info_file.write("\n")

        # Write frequencies
        info_file.write(f"Individual token frequencies:\n")
        info_file.write(f"\n")
        for i, (token, count) in enumerate(most_common):
            info_file.write(f"{i}\t{token}\t{count:,}\n")


def main():

    for corpus_meta in Preferences.source_corpus_metas:

        freq_dist: FreqDist = FreqDist.load(corpus_meta.freq_dist_path)

        info_dir = os.path.dirname(corpus_meta.freq_dist_path)

        save_frequency_distribution_info(
            freq_dist,
            os.path.join(info_dir, f"Frequency distribution info {corpus_meta.name}.txt"))
        save_frequency_distribution_graph(
            freq_dist,
            os.path.join(info_dir, f"Frequency distribution graph {corpus_meta.name}.png"),
            corpus_name=corpus_meta.name,
            top_n=200)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    main()

    logger.info("Done!")
