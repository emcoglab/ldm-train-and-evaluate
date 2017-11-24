"""
===========================
Manipulation of pandas.DataFrame objects.
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

from pandas import DataFrame, isnull


# Model name transformations

def model_name(r):
    if r["Model category"] == "Predict":
        return f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Window radius']} {r['Embedding size']:.0f}"
    else:
        return f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Window radius']}"


def model_name_without_distance(r):
    if r['Model category'] == "Predict":
        return f"{r['Model type']} {r['Embedding size']:.0f} r={r['Window radius']} {r['Corpus']}"
    else:
        return f"{r['Model type']} r={r['Window radius']} {r['Corpus']}"


def model_name_without_corpus_or_distance_or_radius(r):
    if r['Model category'] == "Predict":
        return f"{r['Model type']} {r['Embedding size']:.0f}"
    else:
        return f"{r['Model type']}"


def model_name_without_radius(r):
    if r['Model category'] == "Predict":
        return f"{r['Model type']} {r['Embedding size']:.0f} {r['Distance type']} {r['Corpus']}"
    else:
        return f"{r['Model type']} {r['Distance type']} {r['Corpus']}"


def model_name_without_embedding_size(r):
    return f"{r['Model type']} r={r['Window radius']} {r['Distance type']} {r['Corpus']}"


# Add columns

def add_model_category_column(df: DataFrame):
    """
    Adds a "Model category" column to a results dataframe, containing either "Count" or "Predict".
    """
    df["Model category"] = df.apply(
        lambda r: "Count" if isnull(r["Embedding size"]) else "Predict", axis=1)


def add_model_name_column(df: DataFrame, name_alteration=model_name):
    """
    Adds a "Model name" column to a results dataframe, based on the content of other columns.
    """
    df["Model name"] = df.apply(name_alteration, axis=1)


# Filtering

def predict_models_only(df: DataFrame) -> DataFrame:
    return df[df["Model category"] == "Predict"]
