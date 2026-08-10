import logging
from deltakit_explorer.analysis.threshold import ThresholdEstimator
from plot_threshold import create_threshold_plot

# Configure the logger to show up nicely in your terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting quantum simulations. This might take a minute...")

# 1. Initialize our engine
estimator = ThresholdEstimator()

# 2. Run the multi-point search
threshold_result = estimator.run_single_pair_search(d_low=3, d_high=5)

# 3. Extract the data history
data_history = estimator.get_search_history() 

logger.info(f"Calculations complete! Estimated Threshold: {threshold_result:.5f}")
logger.info("Generating threshold graph...")

# 4. Draw the graph
create_threshold_plot(data_dict=data_history, estimated_threshold=threshold_result)
