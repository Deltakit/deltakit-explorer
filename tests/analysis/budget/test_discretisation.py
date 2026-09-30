import itertools

import numpy as np
import numpy.typing as npt
import pytest

from deltakit_explorer.analysis.error_budget._discretisation import (
    DiscretisationStrategy,
    GradientFitDiscretisationGenerator,
    get_c_optimal_points,
    get_linear_points,
    get_logarithmic_points,
)


def _assert_is_linear(arr: npt.NDArray[np.floating]) -> None:
    diff = np.abs(arr[1:] - arr[:-1])
    np.testing.assert_allclose(diff - diff[0], 0, atol=1e-7)


class TestDiscretisation:

    @pytest.mark.parametrize(
        ("a", "b", "c", "num_points", "degree"),
        list(itertools.product([-1, 0, 0.1], [1, 2], [0.5], [5, 10, 1000], [1, 2, 3])),
    )
    def test_linear_points(
        self, a: float, b: float, c: float, num_points: int, degree: int
    ) -> None:
        ret = get_linear_points(a, b, c, num_points, degree)
        assert len(ret) == num_points
        assert np.all(np.logical_and(a <= ret, ret <= b))
        _assert_is_linear(ret)


    @pytest.mark.parametrize(
        ("a", "b", "c", "num_points", "degree"),
        list(
            itertools.product([0.1, 0.5, 1.0], [1.1, 2.0, 5.0], [1.05], [5, 10], [1, 2, 3])
        ),
    )
    def test_logarithmic_points(
        self, a: float, b: float, c: float, num_points: int, degree: int
    ) -> None:
        ret = get_logarithmic_points(a, b, c, num_points, degree)
        assert len(ret) == num_points
        eps = 1e-7
        assert np.all(np.logical_and(a <= ret + eps, ret <= b + eps))
        _assert_is_linear(np.log10(ret))

    @pytest.mark.parametrize(
        ("func", "abc"),
        list(
            itertools.product(
                [get_linear_points, get_logarithmic_points, get_c_optimal_points],
                [
                    (1.0, 2.0, 3.0),  # a < b < c
                    (2.0, 1.0, 3.0),  # b < a < c
                    (3.0, 1.0, 2.0),  # b < c < a
                    (2.0, 3.0, 1.0),  # c < a < b
                    (3.0, 2.0, 1.0),  # c < b < a
                ],
            )
        ),
    )
    def test_raises_on_invalid_inputs(
        self, func: GradientFitDiscretisationGenerator, abc: tuple[float, float, float]
    ) -> None:
        """Invalid ``a < c < b`` orderings must raise.

        Args:
            func (GradientFitDiscretisationGenerator): The discretisation generator instance.
            abc (tuple[float, float, float]): The test parameters.
        """
        a, b, c = abc
        with pytest.raises(ValueError, match=f"Expected {a=} < {c=} < {b=}"):
            func(a, b, c, 5, 3)


    def test_raise_on_negative_inputs_log(self) -> None:
        with pytest.raises(
            ValueError,
            match="Cannot get logarithmically-spaced points for negative values.*",
        ):
            get_logarithmic_points(-1, 1, 0, 5, 3)


    # ---------------------------------------------------------------------------
    # C-optimal tests
    # ---------------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("a", "b", "num_points", "degree"),
        list(itertools.product([2e-3, 1e-2], [5e-2, 1e-1], [5, 10, 15], [1, 2, 3])),
    )
    def test_c_optimal_points_shape_and_bounds(
        self, a: float, b: float, num_points: int, degree: int
    ) -> None:
        c = (a + b) / 2
        ret = get_c_optimal_points(a, b, c, num_points, degree)
        assert len(ret) == num_points
        assert np.all(np.logical_and(a <= ret, ret <= b))
        assert np.all(ret[:-1] <= ret[1:])


    def test_c_optimal_raises_below_minimum_num_points(self) -> None:
        with pytest.raises(ValueError, match="must sample at least"):
            get_c_optimal_points(2e-3, 1e-2, 7e-3, 3, 3)


    @pytest.mark.parametrize(
        "bad_c", [np.array([7e-3]), np.array([[7e-3]]), np.array([6e-3, 8e-3])]
    )
    def test_c_optimal_rejects_non_scalar_c(self, bad_c: npt.NDArray[np.floating]) -> None:
        with pytest.raises(ValueError, match="c must be a scalar"):
            get_c_optimal_points(2e-3, 1e-2, bad_c, 10, 3)


    def test_c_optimal_is_deterministic(self) -> None:
        pts1 = get_c_optimal_points(2e-3, 1e-2, 7e-3, 10, 3)
        pts2 = get_c_optimal_points(2e-3, 1e-2, 7e-3, 10, 3)
        np.testing.assert_array_equal(pts1, pts2)


    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (1e-5, 1e-3),  # very small noise regime
            (1e-3, 1e-2),  # default
            (1e-2, 5e-2),  # higher noise regime
        ],
    )
    def test_c_optimal_produces_wellconditioned_designs(self, a: float, b: float) -> None:
        c = (a + b) / 2
        pts = get_c_optimal_points(a, b, c, 10, 3)
        u = 2 * (pts - pts.min()) / (pts.max() - pts.min()) - 1
        X = np.vander(u, 4, increasing=True)
        assert np.linalg.cond(X.T @ X) < 1e10


    @pytest.mark.parametrize(
        ("num_points", "degree"),
        [(2, 1), (3, 2), (4, 3), (5, 4)],
    )
    def test_c_optimal_minimal_design_has_enough_unique_points(
        self, num_points: int, degree: int
    ) -> None:
        pts = get_c_optimal_points(2e-3, 1e-2, 7e-3, num_points, degree)
        assert np.unique(pts).size >= degree + 1


    def test_discretisation_strategy_exposes_c_optimal(self) -> None:
        assert hasattr(DiscretisationStrategy, "C_OPTIMAL")


    @pytest.mark.parametrize(
        ("num_points", "degree"), list(itertools.product([5, 10], [1, 2, 3]))
    )
    def test_discretisation_strategy_dispatches_c_optimal(
        self, num_points: int, degree: int
    ) -> None:
        pts = DiscretisationStrategy.C_OPTIMAL(2e-3, 1e-2, 7e-3, num_points, degree)
        assert len(pts) == num_points
        assert np.all((pts >= 2e-3) & (pts <= 1e-2))
