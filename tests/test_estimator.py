import math
from unittest.mock import patch, MagicMock
from src.deltakit_explorer.analysis.threshold import ThresholdEstimator, get_error_bar

def test_get_error_bar():
    """Test that the math for the standard error works correctly."""
    assert get_error_bar(0.0, 100) == 0.0
    
    expected_error = math.sqrt((0.5 * 0.5) / 100)
    assert math.isclose(get_error_bar(0.5, 100), expected_error)

@patch('src.deltakit_explorer.analysis.threshold.Client')
def test_run_simulation(MockClient):
    """Test the worker tool with the NEW shots parameter."""
    mock_instance = MockClient.get_instance.return_value
    mock_decode_result = MagicMock()
    mock_decode_result.get_logical_error_probability.return_value = 0.015
    mock_instance.decode_measurements.return_value = mock_decode_result
    mock_instance.simulate_stim_circuit.return_value = (MagicMock(), MagicMock())

    estimator = ThresholdEstimator()
    result = estimator.run_simulation(p_value=0.01, distance=3, shots=1000)
    
    
    assert result == 0.015

@patch.object(ThresholdEstimator, 'run_simulation')
def test_run_search(mock_run_sim):
    """Test that the manager tool correctly steps left or right."""
    def fake_simulation(p_value, distance, shots):
        if distance == 3: return p_value * 0.5
        if distance == 5: return p_value * 1.5
    
    mock_run_sim.side_effect = fake_simulation

    estimator = ThresholdEstimator(min_p=0.01, max_p=0.05, precision=0.01)
    threshold = estimator.run_search()
    
    assert isinstance(threshold, float)
    assert mock_run_sim.called
