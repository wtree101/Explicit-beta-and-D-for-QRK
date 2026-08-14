import unittest
from functools import partial

import numpy as np

import qrk_adv.feasibility as compatibility_feasibility
import qrk_adv.upper_bound as compatibility_upper_bound
from experiments import heatmaps
from heatmap_data_generation import heatmapDataGeneration as legacy_heatmaps
from qrk_analysis.core.divergence import DKL
from qrk_analysis.core.quantile import sigma_min_alpha0_square
from qrk_analysis.feasibility.check import (
    check_feasibility_conditions,
    check_feasibility_conditions_C_sup_revised,
    check_feasibility_conditions_random_sup_revised,
)
from qrk_analysis.noise.oblivious import (
    error_increased_Gaussian_noise,
    error_increased_Gaussian_noise_batch,
)
from qrk_analysis.upper_bound import smallest_D


class CanonicalFormulaTests(unittest.TestCase):
    def test_baseline_primitive_values(self):
        self.assertAlmostEqual(DKL(0.2, 0.1), 0.04440300758688234)
        self.assertAlmostEqual(
            sigma_min_alpha0_square(0.5, 0.8),
            0.07132591774425939,
        )

    def test_gaussian_batch_matches_scalar_quadrature(self):
        sigmas = np.array([0.01, 0.1, 1.0, 5.0, 10.0])
        expected = np.array(
            [error_increased_Gaussian_noise(1.0, sigma) for sigma in sigmas]
        )
        actual = error_increased_Gaussian_noise_batch(1.0, sigmas)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0.0)

    def test_revised_checks_use_scaled_interval(self):
        arguments = dict(
            T=1,
            beta=0.05,
            D=np.inf,
            q=0.75,
            alpha_0=0.50,
            alpha_prime=0.10,
            delta_f=1.0,
            num_grid_Q=3,
        )
        fixed = check_feasibility_conditions_C_sup_revised(
            **arguments,
            C_min=0.0,
            C_max=1.0,
            num_points_C=2,
        )
        gaussian = check_feasibility_conditions_random_sup_revised(
            **arguments,
            sigma_min=0.01,
            sigma_max=0.02,
            num_points_C=2,
        )
        expected_endpoints = np.array(
            [
                arguments["alpha_0"] / (1.0 - arguments["beta"]),
                1.0
                - arguments["alpha_prime"] / (1.0 - arguments["beta"]),
            ]
        )
        for result in (fixed, gaussian):
            np.testing.assert_array_equal(
                result["Qq_grid"][[0, -1]],
                expected_endpoints,
            )

    def test_failure_constraint_can_be_diagnostic_only(self):
        arguments = dict(
            T=20_000,
            beta=0.005,
            D=1,
            q=0.8,
            alpha_0=0.1,
            alpha_prime=0.19,
            delta_f=0.1,
            c_target=-np.inf,
        )
        enforced = check_feasibility_conditions(**arguments)
        diagnostic = check_feasibility_conditions(
            **arguments,
            enforce_failure_probability=False,
        )
        self.assertFalse(enforced["feasible"])
        self.assertTrue(diagnostic["feasible"])
        self.assertFalse(diagnostic["failure_constraint_enforced"])


class IntegerSearchTests(unittest.TestCase):
    def test_paper_representative_values(self):
        massart = smallest_D(0.01, 20_000, 0.75, 0.1, D_max=500, num_grid=50)
        fixed_check = partial(
            check_feasibility_conditions_C_sup_revised,
            num_grid_Q=10,
            C_min=0.0,
            C_max=20.0,
            num_points_C=200,
        )
        oblivious = smallest_D(
            0.01,
            20_000,
            0.75,
            0.1,
            D_max=1_000,
            num_grid=60,
            feasibility_check=fixed_check,
        )
        self.assertEqual(massart["smallest_D"], 25)
        self.assertEqual(oblivious["smallest_D"], 13)
        for result in (massart, oblivious):
            self.assertIsInstance(result["smallest_D"], int)
            self.assertGreater(result["alpha_0"], 0.0)
            self.assertGreater(result["alpha_prime"], 0.0)
            self.assertLess(result["alpha_0"], 0.75 - 0.01)
            self.assertLess(result["alpha_prime"], 1.0 - 0.75 - 0.01)

    def test_ceiling_diagnostics(self):
        result = smallest_D(0.01, 20_000, 0.75, 0.1, D_max=25)
        exhausted = smallest_D(0.01, 20_000, 0.75, 0.1, D_max=24)
        self.assertTrue(result["hit_ceiling"])
        self.assertEqual(result["smallest_D"], 25)
        self.assertIsNone(exhausted["smallest_D"])
        self.assertTrue(exhausted["search_exhausted"])


class CompatibilityTests(unittest.TestCase):
    def test_qrk_adv_forwards_to_canonical_objects(self):
        self.assertIs(
            compatibility_feasibility.check_feasibility,
            check_feasibility_conditions,
        )
        self.assertIs(compatibility_upper_bound.smallest_D, smallest_D)

    def test_heatmap_facade_forwards_to_reorganized_package(self):
        self.assertIs(
            legacy_heatmaps.streaming_subsampled_qRK_step,
            heatmaps.streaming_subsampled_qRK_step,
        )
        self.assertIs(
            legacy_heatmaps.generate_heat_map_matrix,
            heatmaps.generate_heat_map_matrix,
        )


if __name__ == "__main__":
    unittest.main()
