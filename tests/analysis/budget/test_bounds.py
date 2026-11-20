# (c) Copyright Riverlane 2020-2026. All rights reserved.
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from deltakit_explorer.analysis.error_budget import (
    BoundsDiscoveryError,
    BoundSearchParameters,
    _budget,
    _lambda,
    find_error_budget_bounds,
    get_error_budget,
)

ROUNDS = {3: [2, 4], 5: [2, 4]}


@pytest.fixture
def pilot(monkeypatch):
    calls = []

    def sample(model, parameters, rounds, sampling, _memory):
        x = float(parameters[0])
        calls.append(x)
        model(Mock(), np.array([x]))
        rows = []
        for d, rs in rounds.items():
            rate = 0.03 * (0.25 + 20 * x) ** ((d - 3) / 2)
            for r in rs:
                lep = (1 - (1 - 2 * rate) ** r) / 2
                rows.append(
                    (d, r, round(lep * sampling.max_shots), sampling.max_shots, x)
                )
        return (
            np.array([[x]]),
            ["0"],
            pd.DataFrame(
                rows, columns=["distance", "num_rounds", "fails", "shots", "noise_0"]
            ),
        )

    monkeypatch.setattr(_lambda, "_run_lambda_engine", sample)
    return calls


def config(**kwargs):
    return BoundSearchParameters([(0, 0.05)], shots_per_trial=100_000, **kwargs)


def test_scalar_search_retains_counts_and_estimates(pilot):
    model = Mock()
    result = find_error_budget_bounds(model, 0.02, ROUNDS, search_parameters=config())
    assert result.bounds == pytest.approx((0.0075, 0.0125))
    assert pilot == [0.01, 0.0075, 0.0125]
    assert all(isinstance(call.args[1], float) for call in model.call_args_list)
    assert result.pilot_data.shots.sum() == 1_200_000
    assert result.pilot_data.inverse_lambda.notna().all()
    assert (result.pilot_data.inverse_lambda_stddev > 0).all()


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_discovery_and_budget_use_same_point(pilot, monkeypatch, scale):
    gradient = Mock(return_value=(np.ones(2), np.ones(2)))
    monkeypatch.setattr(_budget, "inverse_lambda_gradient_at", gradient)
    model = Mock()
    result = get_error_budget(
        model,
        [0.02, 0.03],
        ROUNDS,
        None,
        gradient_evaluation_scale=scale,
        bound_search_parameters=BoundSearchParameters(
            [(0, 0.1)] * 2, shots_per_trial=100_000
        ),
    )
    assert pilot
    assert len(result.bound_search_results) == 2
    np.testing.assert_allclose(
        gradient.call_args.args[1], np.array([0.02, 0.03]) * scale
    )
    assert model.call_args_list[1].args[1][1] == pytest.approx(0.03 * scale)
    assert model.call_args_list[-1].args[1][0] == pytest.approx(0.02 * scale)
    assert result.contributions == (0.02, 0.03)


def test_infeasible_endpoints_contract_and_preserve_failed_probes(pilot):
    result = find_error_budget_bounds(
        Mock(),
        0.02,
        ROUNDS,
        search_parameters=config(min_logical_failures=2400),
    )
    assert result.bounds is not None
    assert 0.0075 < result.bounds[0] < 0.01 < result.bounds[1]
    assert result.pilot_data.inverse_lambda.isna().any()
    assert len(pilot) == len(set(pilot))


def test_low_snr_expands_then_stops_at_allowance(pilot):
    result = find_error_budget_bounds(
        Mock(),
        0.02,
        ROUNDS,
        search_parameters=config(sensitivity_z_score=1e9, max_iterations=2),
    )
    assert result.bounds is None
    assert len(pilot) == 5
    assert min(pilot) < 0.0075
    assert max(pilot) > 0.0125


@pytest.mark.parametrize(
    "limits", [{"min_logical_failures": 100_000}, {"max_lep": 0.001}]
)
def test_failed_discovery_prevents_production_and_retains_evidence(
    pilot, monkeypatch, limits
):
    gradient = Mock()
    monkeypatch.setattr(_budget, "inverse_lambda_gradient_at", gradient)
    with pytest.raises(BoundsDiscoveryError) as error:
        get_error_budget(
            Mock(),
            [0.02],
            ROUNDS,
            None,
            bound_search_parameters=config(**limits),
        )
    assert error.value.result.bounds is None
    assert not error.value.result.pilot_data.empty
    gradient.assert_not_called()
    assert pilot == [0.01]


def test_reversed_inverse_lambda_trend_is_rejected(monkeypatch):
    """Reject Filma1's significant decrease in inverse-Lambda as noise increases."""
    calls = []

    def sample(_model, parameters, rounds, sampling, _memory):
        x = float(parameters[0])
        calls.append(x)
        rows = []

        # Keep d=3 fixed while d=5 improves with x, reversing the expected trend.
        for d, rs in rounds.items():
            rate = 0.03 if d == 3 else 0.03 * (0.55 - 15 * x)
            for r in rs:
                lep = (1 - (1 - 2 * rate) ** r) / 2
                rows.append(
                    (d, r, round(lep * sampling.max_shots), sampling.max_shots, x)
                )

        return (
            np.array([[x]]),
            ["0"],
            pd.DataFrame(
                rows, columns=["distance", "num_rounds", "fails", "shots", "noise_0"]
            ),
        )

    monkeypatch.setattr(_lambda, "_run_lambda_engine", sample)
    result = find_error_budget_bounds(
        Mock(),
        0.005,
        ROUNDS,
        search_parameters=BoundSearchParameters(
            [(0, 0.05)], shots_per_trial=100_000, sensitivity_z_score=1.0
        ),
    )

    estimates = (
        result.pilot_data[["noise_0", "inverse_lambda", "inverse_lambda_stddev"]]
        .drop_duplicates()
        .dropna()
        .sort_values("noise_0")
    )
    assert len(estimates) == len(calls) == 5
    initial_low, initial_high = estimates.iloc[1], estimates.iloc[-2]
    assert 0 < initial_low.inverse_lambda - initial_high.inverse_lambda < np.hypot(
        initial_low.inverse_lambda_stddev, initial_high.inverse_lambda_stddev
    )
    low, high = estimates.iloc[0], estimates.iloc[-1]
    assert high.inverse_lambda < low.inverse_lambda
    assert low.inverse_lambda - high.inverse_lambda > np.hypot(
        low.inverse_lambda_stddev, high.inverse_lambda_stddev
    )
    assert result.bounds is None
    assert "decreases significantly" in result.failure_reason
    assert "increasing physical noise strength" in str(BoundsDiscoveryError(result))


def test_explicit_bounds_bypass_pilots(pilot, monkeypatch):
    monkeypatch.setattr(
        _budget,
        "inverse_lambda_gradient_at",
        Mock(return_value=(np.ones(1), np.ones(1))),
    )
    get_error_budget(Mock(), [0.02], ROUNDS, [(0.005, 0.015)])
    assert not pilot


@pytest.mark.parametrize("parameter", [0, [0.02], np.inf, True])
def test_invalid_scalar_fails_before_sampling(pilot, parameter):
    with pytest.raises(ValueError):
        find_error_budget_bounds(Mock(), parameter, ROUNDS, search_parameters=config())
    assert not pilot


@pytest.mark.parametrize("domains", [[(1, 0)], [(0, np.inf)], [], [(0,)]])
def test_invalid_domains(domains):
    with pytest.raises(ValueError):
        BoundSearchParameters(domains)
