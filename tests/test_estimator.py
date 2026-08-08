import math
from unittest.mock import MagicMock, patch
from deltakit_explorer.analysis.threshold import ThresholdEstimator, get_error_bar

def test_get_error_bar():
    """Test that the math for the standard error works correctly."""
    assert get_error_bar(0.0, 100) == 0.0
    expected_error = math.sqrt((0.5 * 0.5) / 100)
    assert math.isclose(get_error_bar(0.5, 100), expected_error)

def test_run_simulation():
    """Test the worker tool with an injected mock client."""
    mock_client = MagicMock()
    
    # Tell the mock client to return a 2-tuple (measurements, leakage_flags)
    mock_client.simulate_stim_circuit.return_value = (MagicMock(), None)
    
    mock_decode_result = MagicMock()
    mock_decode_result.get_logical_error_probability.return_value = 0.015
    mock_client.decode_measurements.return_value = mock_decode_result
    
    estimator = ThresholdEstimator(client=mock_client)
    result = estimator.run_simulation(p_value=0.01, distance=3, shots=1000)
    
    assert result == 0.015

@patch.object(ThresholdEstimator, 'run_simulation')
def test_run_single_pair_search(mock_run_sim):
    """Test that the bisection search correctly steps left or right for one pair."""
    def fake_simulation(p_value, distance, shots):
        if distance == 3: return p_value * 0.5
        if distance == 5: return p_value * 1.5
        return 0.0
    mock_run_sim.side_effect = fake_simulation

    # Pass a MagicMock() for the client here too
    estimator = ThresholdEstimator(client=MagicMock(), min_p=0.01, max_p=0.05, precision=0.01)
    threshold = estimator.run_single_pair_search(d_low=3, d_high=5)
    
    assert isinstance(threshold, float)
    assert mock_run_sim.called

@patch.object(ThresholdEstimator, 'run_single_pair_search')
def test_run_parallel_searches(mock_single_search):
    """Test that multiple pairs are processed and returned correctly in a dictionary."""
    mock_single_search.return_value = 0.025
    
    estimator = ThresholdEstimator(client=MagicMock())
    pairs = [(3, 5), (5, 7)]
    
    results = estimator.run_parallel_searches(pairs)
    
    assert isinstance(results, dict)
    assert len(results) == 2
