import logging
import numpy as np
from scipy.optimize import curve_fit

from deltakit_explorer.analysis.threshold import ThresholdEstimator
from deltakit_explorer.plotting.threshold_plot import create_threshold_plot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def finite_size_scaling_drift(d, p_th, C, omega):
    """
    The physical drift equation for finite-size scaling:
    p_cross(d) = p_th + C * d^(-omega)
   
    Reference:
    Wang, C., Harrington, J., & Preskill, J. (2003). 
    "Confinement-Higgs transition in a disordered gauge theory and the accuracy threshold for quantum memory."
    arXiv:quant-ph/0207088v2

    """
    return p_th + C * (d ** -omega)

def main():
    logger.info("Starting quantum simulations across multiple code distances...")
    
    estimator = ThresholdEstimator()
    
    distance_pairs = [(3, 5), (5, 7), (7, 9)]
    
    logger.info(f"Running bisection searches for pairs: {distance_pairs}.")
    results = estimator.run_parallel_searches(distance_pairs)
    
    # Extract the crossing points and calculate average distances
    average_distances = []
    crossing_points = []
    
    logger.info("--- Pairwise Pseudothreshold Results ---")
    for pair, p_cross in sorted(results.items()):
        d_low, d_high = pair
        d_avg = (d_low + d_high) / 2.0
        
        average_distances.append(d_avg)
        crossing_points.append(p_cross)
        
        logger.info(f"Pair d={pair} (Avg d={d_avg}): p_cross = {p_cross:.5f}")

    # Calculate the True Asymptotic Threshold (p_th) using SciPy
    logger.info("Fitting data to the finite-size scaling drift equation...")
    
    # Initial guesses for the curve fitter: [p_th, C, omega]
    initial_guess = [0.01, 0.01, 1.0]
    
    # Fit the data to the formula
    popt, pcov = curve_fit(
        finite_size_scaling_drift, 
        np.array(average_distances), 
        np.array(crossing_points),
        p0=initial_guess,
        bounds=([0.0, -10.0, 0.1], [0.15, 10.0, 5.0]) # Keeps parameters within realistic  boundaries.
    )
    
    p_th_final, C_final, omega_final = popt
    p_th_error = np.sqrt(np.diag(pcov))[0] # Calculates the statistical uncertainty
    
    logger.info(" Final Results")
    logger.info(f"True Asymptotic Threshold (p_th): {p_th_final:.5f} +/- {p_th_error:.5f}")
    logger.info(f"Extracted Scaling Exponent (omega): {omega_final:.3f}")
    
    #  Extract the data history and draw the graph
    logger.info("Generating threshold graph with the new asymptotic p_th...")
    data_history = estimator.get_search_history() 
    
    create_threshold_plot(
        data_dict=data_history, 
        estimated_threshold=p_th_final,
        threshold_error=p_th_error
    )
# this prevents main() from triggering automatically and running accidental bisection searches.
if __name__ == "__main__":
    main()
