import logging

from deltakit_explorer.analysis.threshold import ThresholdEstimator

# Import the central dispatcher instead of the specific threshold plot
from deltakit_explorer.plotting import plot
from deltakit_explorer.plotting.plotting import ThresholdData

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_experiment() -> None:
    logger.info("Starting quantum simulations across multiple code distances...")

    # SI1000 crossings sit around p~0.004; above ~0.01 every distance saturates
    # at LEP~0.5, so comparisons there are pure noise and mislead the bisection.
    estimator = ThresholdEstimator(max_p=0.01)
    distance_pairs = [(3, 5), (5, 7), (7, 9)]

    logger.info("Running bisection searches for each distance pair...")

    crossing_points = estimator.estimate_crossing_points(distance_pairs)
    crossing_errors = estimator.get_crossing_uncertainties()

    logger.info("--- Pairwise Crossing Points ---")
    for (d_low, d_high), p_cross in sorted(crossing_points.items()):
        logger.info(
            "d = (%d, %d): p_cross = %.2e ± %.0e",
            d_low,
            d_high,
            p_cross,
            crossing_errors[(d_low, d_high)],
        )

    logger.info("Generating threshold graph...")

    fig, _ = plot(
        ThresholdData(
            history_data=estimator.get_search_history(),
            crossing_points=crossing_points,
            crossing_errors=crossing_errors,
        )
    )

    output_filename = "threshold_plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info("Saved threshold plot to %s", output_filename)

if __name__ == "__main__":
    run_experiment()
