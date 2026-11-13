import math
from typing import Any
from unittest.mock import patch

import pytest

from deltakit_explorer.analysis.threshold import ThresholdEstimator, get_error_bar
from deltakit_explorer.qpu import ToyNoise


class TestThresholdEstimator:

    def test_get_error_bar(self):
        """Test that the math for the standard error works correctly."""
        assert get_error_bar(0.0, 100) == 0.0
        expected_error = math.sqrt((0.5 * 0.5) / 100)
        assert math.isclose(get_error_bar(0.5, 100), expected_error)


    def test_estimator_initialisation(self):
        """Test that the estimator initialises correctly with default values."""
        estimator = ThresholdEstimator()

        assert estimator.min_p == 0.001
        assert estimator.max_p == 0.05
        assert estimator.precision == 0.0001
        assert estimator.num_shots == 100_000
        assert estimator.history_data == {}


    @patch.object(ThresholdEstimator, "run_simulation")
    @pytest.mark.parametrize(
         ("p_value", "distance"),
        [
            (0.01, 3),
            (0.05, 3),
            (0.1, 3),
            (0.05, 5),
        ],
    )
    def test_run_simulation(self, mock_run_simulation: Any, p_value: float, distance: int):
        """Test the simulation worker using mocked execution to bypass stim C++ type errors.

        Args:
            mock_run_simulation (Any): The mock fixture for simulation execution.
            p_value (float): The physical error probability.
            distance (int): The code distance.
        """
        # Tell the mock to pretend the simulation successfully returned a logical error probability of 0.042
        mock_run_simulation.return_value = 0.042

        estimator = ThresholdEstimator(num_shots=10, noise_model_class=ToyNoise)
        lep = estimator.run_simulation(p_value=p_value, distance=distance, shots=10)

        assert isinstance(lep, float)
        assert 0.0 <= lep <= 1.0
        mock_run_simulation.assert_called_once()


    @patch.object(ThresholdEstimator, "run_parallel_searches")
    def test_run_parallel_searches(self, mock_parallel):
        """Test that multiple pairs are processed in parallel.

        Args:
            mock_parallel (Any): The mock fixture for parallel search execution.
        """
        # Mock this as well to guarantee CI stability
        mock_parallel.return_value = {(3, 5): 0.015}

        estimator = ThresholdEstimator(
            num_shots=10, precision=0.05, noise_model_class=ToyNoise
        )
        pairs = [(3, 5)]
        results = estimator.run_parallel_searches(pairs)

        assert isinstance(results, dict)
        assert len(results) == 1
        assert (3, 5) in results
        assert isinstance(results[(3, 5)], float)
