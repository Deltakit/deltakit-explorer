import math
from deltakit_explorer.analysis.threshold import ThresholdEstimator, get_error_bar
from deltakit_explorer.qpu import ToyNoise

def test_get_error_bar():
    """Test that the math for the standard error works correctly."""
    assert get_error_bar(0.0, 100) == 0.0
    expected_error = math.sqrt((0.5 * 0.5) / 100)
    assert math.isclose(get_error_bar(0.5, 100), expected_error)

def test_estimator_initialization():
    """Test that the estimator initializes correctly with default values."""
    estimator = ThresholdEstimator()
    
    assert estimator.min_p == 0.001
    assert estimator.max_p == 0.05
    assert estimator.precision == 0.0001
    assert estimator.num_shots == 100_000
    assert estimator.history_data == {}

def test_run_simulation():
    """Test the simulation worker using real execution with ToyNoise."""
    # Use ToyNoise so the QPU noise injection finds all required attributes
    estimator = ThresholdEstimator(num_shots=10, noise_model_class=ToyNoise)
    
    lep = estimator.run_simulation(p_value=0.1, distance=3, shots=10)
    
    assert isinstance(lep, float)
    assert 0.0 <= lep <= 1.0

def test_run_parallel_searches():
    """Test that multiple pairs are processed in parallel using live simulations."""
    estimator = ThresholdEstimator(num_shots=10, precision=0.05, noise_model_class=ToyNoise)
    pairs = [(3, 5)]

    results = estimator.run_parallel_searches(pairs)
    
    assert isinstance(results, dict)
    assert len(results) == 1
    assert (3, 5) in results
    assert isinstance(results[(3, 5)], float)
