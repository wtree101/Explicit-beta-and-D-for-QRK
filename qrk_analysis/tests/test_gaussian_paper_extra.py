from __future__ import annotations

import numpy as np

from qrk_analysis.feasibility.check import (
    check_feasibility_conditions_random_sup_revised,
)
from qrk_analysis.noise.oblivious import (
    error_increased_Gaussian_noise,
    error_increased_Gaussian_noise_batch,
    error_increased_Gaussian_noise_grid,
)
from qrk_analysis.programs import demo_paper_bounds


def test_batch_gaussian_expectation_matches_scalar_quad() -> None:
    sigmas = np.array([0.01, 0.1, 1.0, 5.0, 10.0])
    for quantile_threshold in (0.25, 1.0, 2.0):
        expected = np.array(
            [
                error_increased_Gaussian_noise(quantile_threshold, sigma)
                for sigma in sigmas
            ]
        )
        actual = error_increased_Gaussian_noise_batch(
            quantile_threshold,
            sigmas,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0.0)

    grid = error_increased_Gaussian_noise_grid(
        np.array([0.25, 1.0, 2.0]),
        sigmas,
    )
    expected_grid = np.array(
        [
            [error_increased_Gaussian_noise(qq, sigma) for sigma in sigmas]
            for qq in (0.25, 1.0, 2.0)
        ]
    )
    np.testing.assert_allclose(grid, expected_grid, atol=1e-8, rtol=0.0)


def test_gaussian_supremum_uses_scaled_quantile_interval() -> None:
    beta = 0.05
    alpha_0 = 0.50
    alpha_prime = 0.10
    result = check_feasibility_conditions_random_sup_revised(
        T=1,
        beta=beta,
        D=np.inf,
        q=0.75,
        alpha_0=alpha_0,
        alpha_prime=alpha_prime,
        delta_f=1.0,
        num_grid_Q=3,
        sigma_min=0.01,
        sigma_max=0.02,
        num_points_C=2,
    )

    assert result["Qq_grid"][0] == alpha_0 / (1.0 - beta)
    assert result["Qq_grid"][-1] == 1.0 - alpha_prime / (1.0 - beta)


def test_cli_defaults_to_paper_and_keeps_groups_independent(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        demo_paper_bounds,
        "generate_paper_figures",
        lambda *, recompute: calls.append(("paper", recompute)),
    )
    monkeypatch.setattr(
        demo_paper_bounds,
        "generate_extra_figures",
        lambda *, recompute: calls.append(("extra", recompute)),
    )

    demo_paper_bounds.main([])
    assert calls == [("paper", False)]

    calls.clear()
    demo_paper_bounds.main(["--paper", "--recompute"])
    assert calls == [("paper", True)]

    calls.clear()
    demo_paper_bounds.main(["--extra", "--recompute"])
    assert calls == [("extra", True)]

    calls.clear()
    demo_paper_bounds.main(["--paper", "--extra"])
    assert calls == [("paper", False), ("extra", False)]


def test_recompute_overwrites_selected_cache(tmp_path) -> None:
    x_values = np.array([1.0, 2.0])
    calls = []

    demo_paper_bounds.load_or_compute(
        "curve",
        x_values,
        lambda value: calls.append(value) or value,
        data_dir=tmp_path,
        recompute=False,
    )
    assert calls == [1.0, 2.0]

    calls.clear()
    _, cached = demo_paper_bounds.load_or_compute(
        "curve",
        x_values,
        lambda value: calls.append(value) or 10.0 * value,
        data_dir=tmp_path,
        recompute=False,
    )
    assert calls == []
    np.testing.assert_array_equal(cached, x_values)

    _, recomputed = demo_paper_bounds.load_or_compute(
        "curve",
        x_values,
        lambda value: calls.append(value) or 10.0 * value,
        data_dir=tmp_path,
        recompute=True,
    )
    assert calls == [1.0, 2.0]
    np.testing.assert_array_equal(recomputed, 10.0 * x_values)
