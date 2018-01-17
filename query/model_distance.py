"""
===========================
Compute the distance between two words from a model.
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

from ..core.corpus.indexing import TokenIndexDictionary, FreqDist
from ..core.model.base import DistributionalSemanticModel
from ..core.model.count import LogCoOccurrenceCountModel, CoOccurrenceCountModel, CoOccurrenceProbabilityModel, TokenProbabilityModel, \
    ContextProbabilityModel, ConditionalProbabilityModel, ProbabilityRatioModel, PPMIModel
from ..core.model.predict import CbowModel, SkipGramModel
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences


def main(args):

    corpus_name = args.corpus_name.lower()
    model_name = args.model_name.lower()
    distance = args.distance.lower()
    radius = args.radius
    size = args.size
    word_1 = args.word_1
    word_2 = args.word_2

    # Switch on distanec type
    if distance == "correlation":
        distance_type = DistanceType.correlation
    elif distance == "cosine":
        distance_type = DistanceType.cosine
    elif distance == "euclidean":
        distance_type = DistanceType.Euclidean
    else:
        raise ValueError()

    # Switch on corpus name
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

    token_indices = TokenIndexDictionary.load(corpus_metadata.index_path)
    freq_dist = FreqDist.load(corpus_metadata.freq_dist_path)

    # Switch on model type
    model_type = DistributionalSemanticModel.ModelType.from_slug(model_name)
    if model_type is DistributionalSemanticModel.ModelType.cbow:
        model = CbowModel(corpus_metadata, radius, size)
    elif model_type is DistributionalSemanticModel.ModelType.skip_gram:
        model = SkipGramModel(corpus_metadata, radius, size)
    elif model_type is DistributionalSemanticModel.ModelType.unsummed_cooccurrence:
        # This is too complicated for now, as it involves chirality.
        raise NotImplementedError()
    elif model_type is DistributionalSemanticModel.ModelType.cooccurrence:
        model = CoOccurrenceCountModel(corpus_metadata, radius, token_indices)
    elif model_type is DistributionalSemanticModel.ModelType.log_cooccurrence:
        model = LogCoOccurrenceCountModel(corpus_metadata, radius, token_indices)
    elif model_type is DistributionalSemanticModel.ModelType.cooccurrence_probability:
        model = CoOccurrenceProbabilityModel(corpus_metadata, radius, token_indices, freq_dist)
    elif model_type is DistributionalSemanticModel.ModelType.token_probability:
        model = TokenProbabilityModel(corpus_metadata, radius, token_indices, freq_dist)
    elif model_type is DistributionalSemanticModel.ModelType.context_probability:
        model = ContextProbabilityModel(corpus_metadata, radius, token_indices, freq_dist)
    elif model_type is DistributionalSemanticModel.ModelType.conditional_probability:
        model = ConditionalProbabilityModel(corpus_metadata, radius, token_indices, freq_dist)
    elif model_type is DistributionalSemanticModel.ModelType.probability_ratio:
        model = ProbabilityRatioModel(corpus_metadata, radius, token_indices, freq_dist)
    elif model_type is DistributionalSemanticModel.ModelType.ppmi:
        model = PPMIModel(corpus_metadata, radius, token_indices, freq_dist)
    else:
        raise ValueError()

    model.train()
    distance = model.distance_between(word_1, word_2, distance_type)

    print(f"Distance between '{word_1}' and '{word_2}' is {distance}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search for a word in a corpus.")

    parser.add_argument("corpus_name", type=str, choices={"bbc", "bnc", "ukwac"}, help="The name of the corpus.")
    parser.add_argument("model_name", type=str, choices={t.slug for t in DistributionalSemanticModel.ModelType}, help="The name of the model.")
    parser.add_argument("radius", type=int, choices=set(Preferences.window_radii), help="The window radius.")
    parser.add_argument("size", type=int, choices=set(Preferences.predict_embedding_sizes), help="The embedding size of the predict model. Ignored if not using a predict model, so just put anything.")
    parser.add_argument("distance", type=str, choices={d.name.lower() for d in DistanceType}, help="Distance type.")
    parser.add_argument("word_1", type=str, help="The first word.")
    parser.add_argument("word_2", type=str, help="The second word.")

    main(parser.parse_args())
