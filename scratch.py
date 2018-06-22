from core.corpus.indexing import FreqDistIndex
from .preferences.preferences import Preferences


def convert_picked_freqdists_to_json():

    freq_dists_to_convert = [
        Preferences.bnc_processing_metas['tokenised'].freq_dist_path,
        # Preferences.bnc_text_processing_metas['tokenised'].freq_dist_path,
        # Preferences.bnc_speech_processing_metas['tokenised'].freq_dist_path,
        Preferences.bbc_processing_metas['tokenised'].freq_dist_path,
        Preferences.ukwac_processing_metas['tokenised'].freq_dist_path
    ]

    for freq_dist_path in freq_dists_to_convert:
        fd = FreqDistIndex.load_as_dict(freq_dist_path)
        fd.save_as_json(freq_dist_path + "2")


def main():
    convert_picked_freqdists_to_json()


if __name__ == '__main__':

    main()
