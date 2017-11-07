import os
import glob
import pandas

from .core.evaluation.synonym import SynonymResults
from .core.evaluation.association import AssociationResults
from .preferences.preferences import Preferences


def main():

    convert_synonyms()
    convert_associations()

# This has a bug where csvs are saved into the same directory they are searched for in,
# so if run twice it will do weird things the second time.
# But it's one-use code, so I'm not going to fix it.


def convert_synonyms():
    results_dir = Preferences.synonym_results_dir
    separator = ","
    header_filename = os.path.join(results_dir, " header.csv")
    data_filenames = glob.glob(os.path.join(results_dir, "*.csv"))
    data_filenames.remove(header_filename)
    with open(os.path.join(results_dir, " header.csv"), mode="r", encoding="utf-8") as header_file:
        column_names = header_file.read().strip().split(separator)
    results_df = pandas.DataFrame(columns=column_names)
    for data_filename in data_filenames:
        partial_df = pandas.read_csv(data_filename, sep=separator, names=column_names,
                                     # Convert percent strings to floats
                                     converters={"Score": lambda val: float(val.strip("%"))/100})
        results_df = results_df.append(partial_df, ignore_index=True)
    new_results = SynonymResults()
    new_results.data = results_df
    assert set(new_results.column_names) == set(results_df.columns.values)
    new_results.save()


def convert_associations():
    results_dir = Preferences.association_results_dir
    separator = ","
    header_filename = os.path.join(results_dir, " header.csv")
    data_filenames = glob.glob(os.path.join(results_dir, "*.csv"))
    data_filenames.remove(header_filename)
    with open(os.path.join(results_dir, " header.csv"), mode="r", encoding="utf-8") as header_file:
        column_names = header_file.read().strip().split(separator)
    results_df = pandas.DataFrame(columns=column_names)
    for data_filename in data_filenames:
        partial_df = pandas.read_csv(data_filename, sep=separator, names=column_names)
        results_df = results_df.append(partial_df, ignore_index=True)
    new_results = AssociationResults()
    new_results.data = results_df
    assert set(new_results.column_names) == set(results_df.columns.values)
    new_results.save()


if __name__ == '__main__':

    main()
