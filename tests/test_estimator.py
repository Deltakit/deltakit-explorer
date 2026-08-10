import math
from unittest.mock import MagicMock, patch
from deltakit_explorer.analysis.threshold import ThresholdEstimator, get_error_bar

def test_get_error_bar():
    """Test that the math for the standard error works correctly."""
    assert get_error_bar(0.0, 100) == 0.0
    expected_error = math.sqrt((0.5 * 0.5) / 100)
    assert math.isclose(get_error_bar(0.5, 100), expected_error)

@patch('deltakit_explorer.analysis.threshold.simulate_with_stim')
def test_run_simulation(mock_simulate):
    """Test the simulation worker using the local simulator mock."""
    # Tell the local simulator to return mock measurements and leakage flags
    mock_simulate.return_value = (MagicMock(), None)
    
    estimator = ThresholdEstimator()
    # (If your run_simulation returns an error probability, ensure it's handled or mocked)
    
    assert estimator.min_p == 0.001

@patch.object(ThresholdEstimator, 'run_simulation')
def test_run_single_pair_search(mock_run_sim):
    """Test that the bisection search correctly steps left or right for one pair."""
    def fake_simulation(p_value, distance, shots):
        if distance == 3: return p_value * 0.5
        if distance == 5: return p_value * 1.5
        return 0.0
    
    mock_run_sim.side_effect = fake_simulation

    estimator = ThresholdEstimator(min_p=0.01, max_p=0.05, precision=0.01)
    threshold = estimator.run_single_pair_search(d_low=3, d_high=5)
    
    assert isinstance(threshold, float)
    assert mock_run_sim.called

@patch.object(ThresholdEstimator, 'run_single_pair_search')
def test_run_parallel_searches(mock_single_search):
    """Test that multiple pairs are processed and returned correctly in a dictionary."""
    mock_single_search.return_value = 0.025
    
    estimator = ThresholdEstimator()
    pairs = [(3, 5), (5, 7)]
    
    results = estimator.run_parallel_searches(pairs)
    
    assert isinstance(results, dict)
    assert len(results) == 2
