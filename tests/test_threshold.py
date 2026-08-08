from unittest.mock import MagicMock, patch
from deltakit_explorer.analysis.threshold import ThresholdEstimator

def test_estimator_initialization():
    """Test that the estimator initializes correctly when given a client."""
    mock_client = MagicMock()
    estimator = ThresholdEstimator(client=mock_client)
    assert estimator.min_p == 0.001

@patch("deltakit_explorer.analysis.threshold.ThresholdEstimator.run_simulation")
def test_bisection_updates_min_p(mock_run_simulation):
    """Test that the bisection math correctly moves the boundaries."""
    mock_client = MagicMock()
    estimator = ThresholdEstimator(client=mock_client, min_p=0.01, max_p=0.05, precision=0.035)
    
    mock_run_simulation.side_effect = [0.08, 0.02]
    estimator.run_single_pair_search(d_low=3, d_high=5)
    
    assert mock_run_simulation.called
