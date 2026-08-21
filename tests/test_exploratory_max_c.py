import unittest

import numpy as np

from qrk_analysis.feasibility.check import check_feasibility
from qrk_analysis.feasibility.search import find_max_c_without_failure_constraint


def fake_feasibility(
    T,
    beta,
    D,
    q,
    alpha_0,
    alpha_prime,
    delta_f,
    c_target=0.0,
    *,
    enforce_failure_probability=True,
):
    c_value = 1.0 - (alpha_0 - 0.3) ** 2
    failure_prob = 0.9
    failure_satisfied = failure_prob <= delta_f
    return {
        "feasible": c_value >= c_target
        and (failure_satisfied or not enforce_failure_probability),
        "c": c_value,
        "p_l_c": 1.0,
        "p_u": 0.1,
        "failure_prob": failure_prob,
        "failure_constraint_enforced": enforce_failure_probability,
        "failure_constraint_satisfied": failure_satisfied,
    }


class MaxContractionSearchTests(unittest.TestCase):
    def test_uses_max_alpha_prime_and_maximizes_alpha_0_grid(self):
        pair, result = find_max_c_without_failure_constraint(
            T=100,
            beta=0.1,
            D=5,
            q=0.8,
            delta_f=0.01,
            num_grid=8,
            feasibility_check=fake_feasibility,
        )

        self.assertAlmostEqual(pair[0], 0.3)
        self.assertAlmostEqual(pair[1], 1.0 - 0.8 - 0.1)
        self.assertAlmostEqual(result["c"], 1.0)
        self.assertFalse(result["failure_constraint_enforced"])
        self.assertFalse(result["failure_constraint_satisfied"])

    def test_selected_c_is_independent_of_T_and_delta_f(self):
        first = find_max_c_without_failure_constraint(
            10, 0.1, 5, 0.8, 0.5, num_grid=8,
            feasibility_check=fake_feasibility,
        )
        second = find_max_c_without_failure_constraint(
            1_000_000, 0.1, 5, 0.8, 1e-12, num_grid=8,
            feasibility_check=fake_feasibility,
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1]["c"], second[1]["c"])


class FailureConstraintTests(unittest.TestCase):
    def test_default_enforces_failure_probability(self):
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

        enforced = check_feasibility(**arguments)
        ignored = check_feasibility(
            **arguments,
            enforce_failure_probability=False,
        )

        self.assertFalse(enforced["feasible"])
        self.assertEqual(enforced["reason"], "failure probability too high")
        self.assertTrue(ignored["feasible"])
        self.assertGreater(ignored["failure_prob"], arguments["delta_f"])
        self.assertFalse(ignored["failure_constraint_enforced"])

if __name__ == "__main__":
    unittest.main()
