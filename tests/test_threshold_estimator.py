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
    def test_run_simulation(
        self, mock_run_simulation: Any, p_value: float, distance: int
    ):
        """Test the simulation worker using mocked execution to bypass stim C++ type errors.

        Args:
            mock_run_simulation: The mock fixture for simulation execution.
            p_value: The physical error probability.
            distance: The code distance.
        """
        # Tell the mock to pretend the simulation successfully returned a logical error probability of 0.042
        mock_run_simulation.return_value = 0.042

        estimator = ThresholdEstimator(num_shots=10, noise_model_class=ToyNoise)
        lep = estimator.run_simulation(p_value=p_value, distance=distance, shots=10)

        assert isinstance(lep, float)
        assert 0.0 <= lep <= 1.0
        mock_run_simulation.assert_called_once()

    @patch.object(ThresholdEstimator, "estimate_crossing_points")
    def test_estimate_crossing_points(self, mock_crossings: Any) -> None:
        """Test that crossing points are returned per distance pair.

        Args:
            mock_crossings: The mock fixture for the crossing-point search.
        """
        # Mock this as well to guarantee CI stability
        mock_crossings.return_value = {(3, 5): 0.015}

        estimator = ThresholdEstimator(
            num_shots=10, precision=0.05, noise_model_class=ToyNoise
        )
        pairs = [(3, 5)]
        results = estimator.estimate_crossing_points(pairs)

        assert isinstance(results, dict)
        assert len(results) == 1
        assert (3, 5) in results
        assert isinstance(results[(3, 5)], float)

    @patch.object(ThresholdEstimator, "run_simulation")
    def test_crossing_uncertainty_is_half_final_bracket(
        self, mock_run_simulation: Any
    ) -> None:
        """Test that the bisection records its final bracket as the uncertainty.

        Args:
            mock_run_simulation: The mock fixture for the simulation step.
        """
        p_cross = 0.004

        # Larger distance has a steeper LEP curve, so the curves cross at p_cross.
        def fake_lep(p: float, distance: int, **_: int) -> float:
            return 0.05 * (p / p_cross) ** (2 if distance == 5 else 1)

        mock_run_simulation.side_effect = fake_lep
        estimator = ThresholdEstimator(
            min_p=0.001, max_p=0.01, precision=0.0001, num_shots=10**12
        )

        crossing = estimator.run_single_pair_search(3, 5)
        p_min, p_max = estimator.crossing_brackets[(3, 5)]
        uncertainty = estimator.get_crossing_uncertainties()[(3, 5)]

        assert p_min <= p_cross <= p_max
        assert p_max - p_min <= estimator.precision
        assert uncertainty == pytest.approx((p_max - p_min) / 2)
        assert abs(crossing - p_cross) <= uncertainty

    @patch.object(ThresholdEstimator, "run_simulation")
    def test_early_stop_uncertainty_uses_gap_slope(
        self, mock_run_simulation: Any
    ) -> None:
        """Test that an early stop narrows the bracket using the gap's slope.

        Args:
            mock_run_simulation: The mock fixture for the simulation step.
        """
        # Bisection midpoints on [0.001, 0.01]: 0.0055, 0.00325, then 0.004375.
        # Below the crossing d=3 is worse; the curves are indistinguishable at
        # the third midpoint, which forces an early stop there.
        leps = {
            (0.0055, 3): 0.08,
            (0.0055, 5): 0.10,
            (0.00325, 3): 0.04,
            (0.00325, 5): 0.03,
            (0.004375, 3): 0.05,
            (0.004375, 5): 0.05,
        }

        def fake_lep(p: float, distance: int, **_: int) -> float:
            return leps[(round(p, 6), distance)]

        mock_run_simulation.side_effect = fake_lep
        estimator = ThresholdEstimator(
            min_p=0.001, max_p=0.01, precision=0.0001, num_shots=10_000
        )

        crossing = estimator.run_single_pair_search(3, 5)
        uncertainty = estimator.get_crossing_uncertainties()[(3, 5)]

        # Gap goes from +0.01 at 0.00325 to -0.02 at 0.0055: slope 0.03 / 0.00225.
        slope = 0.03 / 0.00225
        combined_error = 2 * get_error_bar(0.05, 1_000_000)
        assert crossing == pytest.approx(0.004375)
        assert uncertainty == pytest.approx(combined_error / slope)
        assert uncertainty < (0.0055 - 0.00325) / 2
