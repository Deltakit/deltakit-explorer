from unittest.mock import patch
from deltakit_explorer.analysis.threshold import ThresholdEstimator

def test_estimator_initialization():
    """Test that the estimator saves our starting variables correctly."""
    estimator = ThresholdEstimator(min_p=0.01, max_p=0.05, precision=0.001)
    
    assert estimator.min_p == 0.01
    assert estimator.max_p == 0.05
    assert estimator.precision == 0.001

@patch("deltakit_explorer.analysis.threshold.ThresholdEstimator.run_simulation")
def test_bisection_updates_min_p(mock_run_simulation):
    """Test that the bisection math correctly moves the bottom boundary up."""
    
    estimator = ThresholdEstimator(min_p=0.01, max_p=0.05, precision=0.035)
    
    
    mock_run_simulation.side_effect = [0.08, 0.02]
    
    
    estimator.run_search()
    
     
     
    
    assert estimator.min_p == 0.03
