import seaborn

from .core.corpus.indexing import FreqDist
from .core.model.count import PMIModel
from .preferences.preferences import Preferences


def main():

    meta = Preferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(meta.freq_dist_path)
    radius = 5
    model = PMIModel(meta, radius, freq_dist)
    model.train()

    plot = seaborn.distplot(model.matrix.data)

    plot.figure.savefig("/Users/caiwingfield/Desktop/plot")


if __name__ == '__main__':

    main()
