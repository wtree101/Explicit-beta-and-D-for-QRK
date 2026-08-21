import unittest
from unittest.mock import call, patch

import numpy as np

from experiments.heatmaps import (
    streaming_subsampled_qRK_step,
)


class ObliviousLargeNoiseTests(unittest.TestCase):
    def run_step(
        self,
        corruption_indicators,
        quantile_noise_min=1e16,
        quantile_noise_max=1e16,
        update_noise_min=1e8,
        update_noise_max=1e8,
    ):
        rows = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        with (
            patch("numpy.random.normal", return_value=rows),
            patch(
                "numpy.random.binomial",
                return_value=np.asarray(corruption_indicators),
            ),
        ):
            return streaming_subsampled_qRK_step(
                x=np.array([1.0, 1.0]),
                xk=np.zeros(2),
                q=0.8,
                beta=0.1,
                D=4,
                corruption_type="oblivious_large",
                c_min=0.0,
                c_max=0.0,
                s_min=0.0,
                s_max=0.0,
                quantile_noise_min=quantile_noise_min,
                quantile_noise_max=quantile_noise_max,
                update_noise_min=update_noise_min,
                update_noise_max=update_noise_max,
            )

    def test_large_quantile_noise_allows_corrupted_update(self):
        xk, _ = self.run_step([1, 1, 1, 1, 1])

        self.assertGreater(xk[0], 1e7)
        self.assertTrue(np.all(np.isfinite(xk)))

    def test_clean_quantile_sample_rejects_corrupted_update(self):
        xk, _ = self.run_step([1, 0, 0, 0, 0])

        np.testing.assert_array_equal(xk, np.zeros(2))

    @patch(
        "numpy.random.uniform",
        side_effect=[np.full(4, 1e16), -500.0],
    )
    def test_update_noise_supports_uniform_interval(self, mock_uniform):
        xk, _ = self.run_step(
            [1, 1, 1, 1, 1],
            update_noise_min=-1000.0,
            update_noise_max=1000.0,
        )

        self.assertEqual(
            mock_uniform.call_args_list[1],
            call(low=-1000.0, high=1000.0),
        )
        self.assertLess(xk[0], -100.0)

    @patch(
        "numpy.random.uniform",
        side_effect=[np.array([-1000.0, -500.0, 500.0, 1000.0]), 100.0],
    )
    def test_quantile_noise_supports_uniform_interval(self, mock_uniform):
        self.run_step(
            [0, 1, 1, 1, 1],
            quantile_noise_min=-1000.0,
            quantile_noise_max=1000.0,
        )

        self.assertEqual(
            mock_uniform.call_args_list[0],
            call(low=-1000.0, high=1000.0, size=4),
        )

    def test_rejects_reversed_update_noise_interval(self):
        with self.assertRaises(ValueError):
            self.run_step(
                [1, 1, 1, 1, 1],
                update_noise_min=1000.0,
                update_noise_max=-1000.0,
            )

    def test_rejects_reversed_quantile_noise_interval(self):
        with self.assertRaises(ValueError):
            self.run_step(
                [1, 1, 1, 1, 1],
                quantile_noise_min=1000.0,
                quantile_noise_max=-1000.0,
            )


if __name__ == "__main__":
    unittest.main()
