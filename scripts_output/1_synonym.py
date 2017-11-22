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
from matplotlib import pyplot
from pandas import DataFrame

from ..preferences.preferences import Preferences
from ..core.evaluation.synonym import ToeflTest, LbmMcqTest, EslTest, SynonymResults
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType
from ..core.output.dataframe import add_model_category_column, add_model_name_column
from ..core.output.figures import cosine_vs_correlation_scores

logger = logging.getLogger(__name__)

TEST_NAMES = [ToeflTest().name, EslTest().name, LbmMcqTest().name]

figures_base_dir = os.path.join(Preferences.figures_dir, "synonym")


def main():
    results_df = SynonymResults().load().data

    add_model_category_column(results_df)
    add_model_name_column(results_df)

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

    cosine_vs_correlation_scores(results_df, figures_base_dir, TEST_NAMES, "Score", "Synonym", ticks_as_percentages=True)


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


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
