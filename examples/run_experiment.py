import logging

from deltakit_explorer.analysis.threshold import ThresholdEstimator
# Import the central dispatcher instead of the specific threshold plot
from deltakit_explorer.plotting import plot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_experiment() -> None:
    logger.info("Starting quantum simulations across multiple code distances...")

    estimator = ThresholdEstimator()
    distance_pairs = [(3, 5), (5, 7), (7, 9)]

    logger.info("Running bisection searches and calculating asymptotic threshold...")

    p_th_final, p_th_error, omega_final, results = estimator.estimate_asymptotic_threshold(distance_pairs)

    logger.info("--- Pairwise Pseudothreshold Results ---")
    for pair, p_cross in sorted(results.items()):
        d_avg = sum(pair) / 2.0
        logger.info("Pair d=%s (Avg d=%s): p_cross = %.5f", pair, d_avg, p_cross)

    logger.info("--- Final Results ---")
    logger.info("True Asymptotic Threshold (p_th): %.5f +/- %.5f", p_th_final, p_th_error)
    logger.info("Extracted Scaling Exponent (omega): %.3f", omega_final)

    logger.info("Generating threshold graph with the new asymptotic p_th...")

    fig, ax = plot(
        result=estimator.get_search_history(),
        estimated_threshold=p_th_final,
        threshold_error=p_th_error
    )

    output_filename = "threshold_plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    logger.info("Saved threshold plot to %s", output_filename)

if __name__ == "__main__":
    run_experiment()
