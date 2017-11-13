"""
===========================
Figures for synonym tests.
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

import seaborn
import pandas
from pandas import DataFrame

from matplotlib import pyplot

from ..core.evaluation.synonym import ToeflTest, LbmMcqTest, EslTest, SynonymResults
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)

TEST_NAMES = [ToeflTest().name, EslTest().name, LbmMcqTest().name]

figures_base_dir = os.path.join(Preferences.figures_dir, "synonym")


# TODO: essentially duplicated code
def main():
    results_df = SynonymResults().load().data

    results_df["Model category"] = results_df.apply(lambda r: "Count" if pandas.isnull(r["Embedding size"]) else "Predict", axis=1)

    results_df["Model name"] = results_df.apply(
        lambda r:
        f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Radius']} {r['Embedding size']:.0f}"
        if r["Model category"] == "Predict"
        else f"{r['Corpus']} {r['Distance type']} {r['Model type']} r={r['Radius']}",
        axis=1
    )

    logger.info(f"Making score-vs-radius figures")
    figures_score_vs_radius(results_df)
    
    for test_name in TEST_NAMES:
        logger.info(f"Making embedding size figures for {test_name}")
        figures_embedding_size(results_df, test_name)

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            logger.info(f"Making model bar graph figures for r={radius} and d={distance_type.name}")
            model_performance_bar_graphs(results_df, window_radius=radius, distance_type=distance_type)

    # Summary tables
    logger.info("Making top-5 model tables overall")
    table_top_n_models(results_df, 5)
    for distance_type in DistanceType:
        logger.info(f"Making top-5 model tables overall for {distance_type.name}")
        table_top_n_models(results_df, 5, distance_type)

    cos_vs_cor_scores(results_df)


def table_top_n_models(regression_results_df: DataFrame, top_n: int, distance_type: DistanceType = None):

    summary_dir = Preferences.summary_dir

    results_df = DataFrame()

    for test_name in TEST_NAMES:

        filtered_df: DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["Test name"] == test_name]

        if distance_type is not None:
            filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

        top_models = filtered_df.sort_values("Score", ascending=False).reset_index(drop=True).head(top_n)

        results_df = results_df.append(top_models)

    if distance_type is None:
        file_name = f"synonym_top_{top_n}_models.csv"
    else:
        file_name = f"synonym_top_{top_n}_models_{distance_type.name}.csv"

    results_df.to_csv(os.path.join(summary_dir, file_name), index=False)


def model_performance_bar_graphs(synonym_results_df: DataFrame, window_radius: int, distance_type: DistanceType):

    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    seaborn.set_style("ticks")

    filtered_df: DataFrame = synonym_results_df.copy()
    filtered_df = filtered_df[filtered_df["Radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["Distance type"] == distance_type.name]

    # Don't want to show PPMI (10000)
    filtered_df = filtered_df[filtered_df["Model type"] != "PPMI (10000)"]

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["Model name"] = filtered_df.apply(
        lambda r:
        f"{r['Model type']} {r['Embedding size']:.0f}"
        if r['Model category'] == "Predict"
        else f"{r['Model type']}",
        axis=1
    )

    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(
        filtered_df,
        row="Test name", col="Corpus",
        margin_titles=True,
        size=2.5,
        ylim=(0, 1))

    grid.set_xticklabels(rotation=-90)

    ytick_labels = grid.axes[0][0].get_yticklabels()
    grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])

    # Plot the bars
    plot = grid.map(seaborn.barplot, "Model name", "Score", order=[
        "log n-gram",
        "Conditional probability",
        "Probability ratio",
        "PPMI",
        "Skip-gram 50",
        "Skip-gram 100",
        "Skip-gram 200",
        "Skip-gram 300",
        "Skip-gram 500",
        "CBOW 50",
        "CBOW 100",
        "CBOW 200",
        "CBOW 300",
        "CBOW 500",
    ])

    # TODO: this isn't working for some reason
    # Remove the "corpus = " from the titles
    grid.set_titles(col_template='{col_name}', row_template="{row_name}")

    # Plot the chance line
    grid.map(pyplot.axhline, y=0.25, linestyle="solid", color="xkcd:bright red")

    grid.set_ylabels("Score")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model scores for radius {window_radius} using {distance_type.name} distance")

    figure_name = f"synonym r={window_radius} {distance_type.name}.png"

    # I don't know why PyCharm doesn't find this... it works...
    # noinspection PyUnresolvedReferences
    plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)

    pyplot.close(grid.fig)


def figures_score_vs_radius(regression_results_df: DataFrame):

    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    for distance_type in [d.name for d in DistanceType]:
        filtered_df: DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["Distance type"] == distance_type]

        # Don't need corpus, radius or distance, as they're fixed for each plot
        filtered_df["Model name"] = filtered_df.apply(
            lambda r:
            f"{r['Model type']} {r['Embedding size']:.0f}"
            if r["Model category"] == "Predict"
            else f"{r['Model type']}",
            axis=1
        )

        # We don't want this one
        filtered_df = filtered_df[filtered_df["Model name"] != "PPMI (10000)"]

        filtered_df = filtered_df.sort_values(by=["Model name", "Radius"], ascending=True)
        filtered_df = filtered_df.reset_index(drop=True)

        seaborn.set_style("ticks")
        seaborn.set_context(context="paper", font_scale=1)
        grid = seaborn.FacetGrid(
            data=filtered_df,
            row="Test name", col="Corpus", hue="Model name",
            hue_order=[
                "log n-gram",
                "Conditional probability",
                "Probability ratio",
                "PPMI",
                "Skip-gram 50",
                "Skip-gram 100",
                "Skip-gram 200",
                "Skip-gram 300",
                "Skip-gram 500",
                "CBOW 50",
                "CBOW 100",
                "CBOW 200",
                "CBOW 300",
                "CBOW 500"
            ],
            palette=[
                "orange",
                "turquoise",
                "pink",
                "red",
                "#0000ff",
                "#2a2aff",
                "#5454ff",
                "#7e7eff",
                "#a8a8ff",
                "#00ff00",
                "#2aff2a",
                "#54ff54",
                "#7eff7e",
                "#a8ffa8",
            ],
            hue_kws=dict(
                marker=[
                    "o",
                    "o",
                    "o",
                    "o",
                    "^",
                    "^",
                    "^",
                    "^",
                    "^",
                    "^",
                    "^",
                    "^",
                    "^",
                    "^",
                ]
            ),
            margin_titles=True,
            legend_out=True,
            size=3.5,
            ylim=(0, 1))
        grid.map(pyplot.plot, "Radius", "Score")

        # Format yticks as percentages
        grid.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in grid.axes[0][0].get_yticks()])
        grid.add_legend(bbox_to_anchor=(1.15, 0.5))

        figure_name = f"synonym effect of radius {distance_type}.png"
        grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)
        pyplot.close(grid.fig)


def figures_embedding_size(regression_results_df: DataFrame, test_name: str):

    figures_dir = os.path.join(figures_base_dir, "effects of embedding size")

    for distance in [d.name for d in DistanceType]:
        filtered_df: DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["Distance type"] == distance]
        filtered_df = filtered_df[filtered_df["Test name"] == test_name]

        # Remove count models by dropping rows with nan in embedding_size column
        filtered_df = filtered_df.dropna()

        filtered_df = filtered_df[[
            "Corpus",
            "Model type",
            "Embedding size",
            "Radius",
            "Score"]]

        filtered_df = filtered_df.sort_values(by=["Corpus", "Model type", "Embedding size", "Radius"])
        filtered_df = filtered_df.reset_index(drop=True)

        seaborn.set_style("ticks")
        seaborn.set_context(context="paper", font_scale=1)
        grid = seaborn.FacetGrid(
            filtered_df,
            row="Radius", col="Corpus",
            margin_titles=True,
            size=2,
            ylim=(0, 1),
            legend_out=True
        )

        grid.map(pyplot.plot, "Embedding size", "Score", marker="o")

        # Chance line
        grid.map(pyplot.axhline, y=0.25, ls=":", c=".5", label="")

        grid.set(
            xticks=Preferences.predict_embedding_sizes,
        )

        ytick_labels = grid.axes[0][0].get_yticklabels()
        grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])

        grid.set_xlabels("Embedding size")
        grid.set_ylabels("Score")

        grid.add_legend(title="Model", bbox_to_anchor=(1, 1))
        # pyplot.legend(loc="lower center")

        # Title
        title = f"{test_name} ({distance})"
        pyplot.subplots_adjust(top=0.92)
        grid.fig.suptitle(title)

        figure_name = f"synonym Embedding size {test_name} {distance}.png"

        grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)

        pyplot.close(grid.fig)


def cos_vs_cor_scores(results_df: DataFrame):

    figures_dir = os.path.join(figures_base_dir, "effects of distance")
    seaborn.set(style="white", palette="muted", color_codes=True)

    distribution = []
    for test_name in TEST_NAMES:

        filtered_df: DataFrame = results_df.copy()
        filtered_df = filtered_df[filtered_df["Test name"] == test_name]

        filtered_df["Model name"] = filtered_df.apply(
            lambda r:
            f"{r['Model type']} {r['Embedding size']:.0f} r={r['Radius']} {r['Corpus']}"
            if r['Embedding size'] is not None
            else f"{r['Model type']} r={r['Radius']} {r['Corpus']}",
            axis=1
        )

        for model_name in set(filtered_df["Model name"]):
            cos_df: DataFrame = filtered_df.copy()
            cos_df = cos_df[cos_df["Model name"] == model_name]
            cos_df = cos_df[cos_df["Distance type"] == "cosine"]

            corr_df: DataFrame = filtered_df.copy()
            corr_df = corr_df[corr_df["Model name"] == model_name]
            corr_df = corr_df[corr_df["Distance type"] == "correlation"]

            # barf
            score_cos = list(cos_df["Score"])[0]
            score_corr = list(corr_df["Score"])[0]

            distribution.append([test_name, score_cos, score_corr])

    dist_df = DataFrame(distribution, columns=["Test name", "Cosine Score", "Correlation Score"])

    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.FacetGrid(data=dist_df, col="Test name", col_wrap=2, size=5)
    grid.set(xlim=(0, 1), ylim=(0, 1))

    grid.map(pyplot.scatter, "Cosine Score", "Correlation Score")

    for ax in grid.axes.flat:
        ax.plot((0, 1), (0, 1), c="r", ls="-")

    # Format xicks as percentages
    xtick_labels = grid.axes[0].get_xticklabels()
    grid.set_xticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in xtick_labels])
    ytick_labels = grid.axes[0].get_yticklabels()
    grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Synonyms: correlation- & cosine-distance Scores")

    grid.savefig(os.path.join(figures_dir, f"synonym Scores cos-vs-cor.png"), dpi=300)
    pyplot.close(grid.fig)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
