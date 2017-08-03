import argparse
import logging
import os
import pickle
import sys

import nltk

from ..core.corpus.corpus import CorpusMetadata, BatchedCorpus
from ..core.corpus.distribution import FreqDistConstructor
from ..core.corpus.filtering import filter_punctuation
from ..core.corpus.tokenising import modified_word_tokenize

logger = logging.getLogger(__name__)


# TODO: Make this into numbered script, referencing preferences
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
        info_file.write(f"Vocabulary size: {len(most_common):,}\n")

        # Write low-frequency counts
        for cutoff_freq in [0, 1, 5, 10, 25, 50, 100]:
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


def save_frequency_distribution(freq_dist, filename):
    with open(filename, mode="wb") as file:
        # TODO: Don't use pickle
        pickle.dump(freq_dist, file)


def main(corpus_path, output_dir, tokenised):
    corpus_name = os.path.basename(corpus_path)

    logger.info(f"Loading corpus documents from {corpus_path}")
    if not tokenised:
        # Use CorpusReader and tokenise
        corpus = nltk.corpus.PlaintextCorpusReader(corpus_path, ".+\..+")

        logger.info(f"Tokenising corpus")
        corpus = [w.lower() for w in modified_word_tokenize(corpus.raw())]

        logger.info(f"Filtering corpus")
        corpus = filter_punctuation(corpus)

        freq_dist = nltk.probability.FreqDist(corpus)

    else:  # tokenised
        freq_dist = FreqDistConstructor.from_batched_corpus(
            BatchedCorpus(CorpusMetadata(path=corpus_path, name=""), batch_size=1_000_000))

    logger.info(f"Saving frequency distribution information")
    save_frequency_distribution_info(
        freq_dist,
        os.path.join(output_dir, f"Frequency distribution info {corpus_name}.txt"))
    save_frequency_distribution_graph(
        freq_dist,
        os.path.join(output_dir, f"Frequency distribution graph {corpus_name}.png"),
        corpus_name=corpus_name,
        top_n=200)
    save_frequency_distribution(
        freq_dist,
        os.path.join(output_dir, f"Frequency distribution {corpus_name}.pickle"))


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenised", action="store_true")
    parser.add_argument("--corpuspath")
    parser.add_argument("--outdir")
    args = vars(parser.parse_args())

    main(
        corpus_path=args["corpuspath"],
        output_dir=args["outdir"],
        tokenised=args["tokenised"])

    logger.info("Done!")
