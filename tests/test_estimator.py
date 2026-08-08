import math
from unittest.mock import MagicMock, patch
from deltakit_explorer.analysis.threshold import ThresholdEstimator, get_error_bar

def test_get_error_bar():
    """Test that the math for the standard error works correctly."""
    assert get_error_bar(0.0, 100) == 0.0
    
    expected_error = math.sqrt((0.5 * 0.5) / 100)
    assert math.isclose(get_error_bar(0.5, 100), expected_error)

@patch('src.deltakit_explorer.analysis.threshold.simulate_with_stim')
def test_run_simulation(mock_simulate_with_stim):
    """Test the worker tool with the NEW shots parameter."""
    mock_result = MagicMock()
    mock_result.get_logical_error_probability.return_value = 0.015
    mock_simulate_with_stim.return_value = mock_result
    estimator = ThresholdEstimator()
    result = estimator.run_simulation(p_value=0.01, distance=3, shots=1000)
    
    assert result == 0.015

@patch.object(ThresholdEstimator, 'run_simulation')
def test_run_single_pair_search(mock_run_sim):
    """Test that the bisection search correctly steps left or right for one pair."""
    def fake_simulation(p_value, distance, shots):
        # Create an artificial difference so the bisection loop works
        if distance == 3: return p_value * 0.5
        if distance == 5: return p_value * 1.5
        return 0.0
    
    mock_run_sim.side_effect = fake_simulation

    estimator = ThresholdEstimator(min_p=0.01, max_p=0.05, precision=0.01)
    
    # We now pass the two distances (3 and 5) to the newly renamed method!
    threshold = estimator.run_single_pair_search(d_low=3, d_high=5)
    
    assert isinstance(threshold, float)
    assert mock_run_sim.called

@patch.object(ThresholdEstimator, 'run_single_pair_search')
def test_run_parallel_searches(mock_single_search):
    """Test that multiple pairs are processed and returned correctly in a dictionary."""
    # Tell the mock to instantly return a fake threshold of 0.025 for ANY pair it receives
    mock_single_search.return_value = 0.025
    
    estimator = ThresholdEstimator()
    pairs = [(3, 5), (5, 7)]
    
    # Run the parallel method
    results = estimator.run_parallel_searches(pairs)
    
    # Check that we got a dictionary back containing both pairs
    assert isinstance(results, dict)
    assert len(results) == 2
    assert results[(3, 5)] == 0.025
    assert results[(5, 7)] == 0.025
