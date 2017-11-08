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
import math

import numpy
import pandas
import seaborn

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
    results_df = ensure_column_safety(results_df)

    results_df["model_name"] = results_df.apply(
        lambda r:
        f"{r['corpus']} {r['distance_type']} {r['model_type']} {r['embedding_size']}"
        if r['embedding_size'] is not None
        else f"{r['corpus']} {r['distance_type']} {r['model_type']}",
        axis=1
    )

    for test_name in TEST_NAMES:
        logger.info(f"Making score-vs-radius figures for {test_name}")
        figures_score_vs_radius(results_df, test_name)
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


def table_top_n_models(regression_results_df: pandas.DataFrame, top_n: int, distance_type: DistanceType = None):

    summary_dir = Preferences.summary_dir

    results_df = pandas.DataFrame()

    for test_name in TEST_NAMES:

        filtered_df: pandas.DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["test_name"] == test_name]

        if distance_type is not None:
            filtered_df = filtered_df[filtered_df["distance_type"] == distance_type.name]

        top_models = filtered_df.sort_values("score", ascending=False).reset_index(drop=True).head(top_n)

        results_df = results_df.append(top_models)

    if distance_type is None:
        file_name = f"synonym_top_{top_n}_models.csv"
    else:
        file_name = f"synonym_top_{top_n}_models_{distance_type.name}.csv"

    results_df.to_csv(os.path.join(summary_dir, file_name), index=False)


def model_performance_bar_graphs(synonym_results_df: pandas.DataFrame, window_radius: int, distance_type: DistanceType):

    figures_dir = os.path.join(figures_base_dir, "model performance bar graphs")

    seaborn.set_style("ticks")

    filtered_df: pandas.DataFrame = synonym_results_df.copy()
    filtered_df = filtered_df[filtered_df["radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["distance_type"] == distance_type.name]

    # Don't want to show PPMI (10000)
    filtered_df = filtered_df[filtered_df["model_type"] != "PPMI (10000)"]

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["model_name"] = filtered_df.apply(
        lambda r:
        f"{r['model_type']} {r['embedding_size']}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['model_type']}",
        axis=1
    )

    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(
        filtered_df,
        row="test_name", col="corpus",
        margin_titles=True,
        size=2.5,
        ylim=(0, 1))

    grid.set_xticklabels(rotation=-90)

    ytick_labels = grid.axes[0][0].get_yticklabels()
    grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])

    # Plot the bars
    plot = grid.map(seaborn.barplot, "model_name", "score", order=[
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


def figures_score_vs_radius(regression_results_df: pandas.DataFrame, test_name: str):

    figures_dir = os.path.join(figures_base_dir, "effects of radius")

    for distance_type in [d.name for d in DistanceType]:
        for corpus in ["BNC", "BBC", "UKWAC"]:
            filtered_df: pandas.DataFrame = regression_results_df.copy()
            filtered_df = filtered_df[filtered_df["corpus"] == corpus]
            filtered_df = filtered_df[filtered_df["distance_type"] == distance_type]
            filtered_df = filtered_df[filtered_df["test_name"] == test_name]

            filtered_df = filtered_df.sort_values(by=["model_name", "radius"])
            filtered_df = filtered_df.reset_index(drop=True)

            filtered_df = filtered_df[[
                "model_name",
                "radius",
                "score"]]

            plot = seaborn.factorplot(data=filtered_df,
                                      x="radius", y="score",
                                      hue="model_name",
                                      size=7, aspect=1.8,
                                      legend=False)

            plot.set(ylim=(0, 1))

            # Format yticks as percentages
            vals = plot.ax.get_yticks()
            plot.ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])

            # Put the legend out of the figure
            # resize figure box to -> put the legend out of the figure
            plot_box = plot.ax.get_position()  # get position of figure
            plot.ax.set_position([plot_box.x0, plot_box.y0, plot_box.width * 0.75, plot_box.height])  # resize position

            # Put a legend to the right side
            plot.ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

            figure_name = f"synonym {test_name} {corpus} {distance_type}.png"

            plot.savefig(os.path.join(figures_dir, figure_name))

            pyplot.close(plot.fig)


def figures_embedding_size(regression_results_df: pandas.DataFrame, test_name: str):

    figures_dir = os.path.join(figures_base_dir, "effects of embedding size")

    for distance in [d.name for d in DistanceType]:
        filtered_df: pandas.DataFrame = regression_results_df.copy()
        filtered_df = filtered_df[filtered_df["distance_type"] == distance]
        filtered_df = filtered_df[filtered_df["test_name"] == test_name]

        # Remove count models by dropping rows with nan in embedding_size column
        filtered_df = filtered_df.dropna()

        filtered_df = filtered_df[[
            "corpus",
            "model_type",
            "embedding_size",
            "radius",
            "score"]]

        filtered_df = filtered_df.sort_values(by=["corpus", "model_type", "embedding_size", "radius"])
        filtered_df = filtered_df.reset_index(drop=True)

        seaborn.set_style("ticks")
        seaborn.set_context(context="paper", font_scale=1)
        grid = seaborn.FacetGrid(
            filtered_df,
            row="radius", col="corpus", hue="model_type",
            margin_titles=True,
            size=2,
            ylim=(0, 1),
            legend_out=True
        )

        grid.map(pyplot.plot, "embedding_size", "score", marker="o")

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

        figure_name = f"synonym embedding_size {test_name} {distance}.png"

        grid.savefig(os.path.join(figures_dir, figure_name), dpi=300)

        pyplot.close(grid.fig)


def cos_vs_cor_scores(results_df: pandas.DataFrame):

    figures_dir = os.path.join(figures_base_dir, "effects of distance")
    seaborn.set(style="white", palette="muted", color_codes=True)

    distribution = []
    for test_name in TEST_NAMES:

        filtered_df: pandas.DataFrame = results_df.copy()
        filtered_df = filtered_df[filtered_df["test_name"] == test_name]

        filtered_df["model_name"] = filtered_df.apply(
            lambda r:
            f"{r['model_type']} {r['embedding_size']:.0f} r={r['radius']} {r['corpus']}"
            if r['embedding_size'] is not None
            else f"{r['model_type']} r={r['radius']} {r['corpus']}",
            axis=1
        )

        for model_name in set(filtered_df["model_name"]):
            cos_df: pandas.DataFrame = filtered_df.copy()
            cos_df = cos_df[cos_df["model_name"] == model_name]
            cos_df = cos_df[cos_df["distance_type"] == "cosine"]

            corr_df: pandas.DataFrame = filtered_df.copy()
            corr_df = corr_df[corr_df["model_name"] == model_name]
            corr_df = corr_df[corr_df["distance_type"] == "correlation"]

            # barf
            score_cos = list(cos_df["score"])[0]
            score_corr = list(corr_df["score"])[0]

            distribution.append([test_name, score_cos, score_corr])

    dist_df = pandas.DataFrame(distribution, columns=["Test name", "Cosine score", "Correlation score"])

    seaborn.set_context(context="paper", font_scale=1)

    grid = seaborn.FacetGrid(data=dist_df, col="Test name", col_wrap=2, size=5)
    grid.set(xlim=(0, 1), ylim=(0, 1))

    grid.map(pyplot.scatter, "Cosine score", "Correlation score")

    for ax in grid.axes.flat:
        ax.plot((0, 1), (0, 1), c="r", ls="-")

    # Format xicks as percentages
    xtick_labels = grid.axes[0].get_xticklabels()
    grid.set_xticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in xtick_labels])
    ytick_labels = grid.axes[0].get_yticklabels()
    grid.set_yticklabels(['{:3.0f}%'.format(float(label.get_text()) * 100) for label in ytick_labels])

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Synonyms: correlation- & cosine-distance scores")

    grid.savefig(os.path.join(figures_dir, f"synonym scores cos-vs-cor.png"), dpi=300)
    pyplot.close(grid.fig)


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
