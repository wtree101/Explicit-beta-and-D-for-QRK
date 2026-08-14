import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from heatmap_data_generation.heatmapDataGeneration import (
    generate_heat_map_matrix,
    make_feasibility_check,
    run_qRK_subsample_D_vs_beta,
    run_qRK_subsample_D_vs_T,
    save_heat_map_matrix,
)


class FixedSuccessCriterionTests(unittest.TestCase):
    def test_oblivious_large_uses_independent_fixed_C_supremum_range(self):
        feasibility_check = make_feasibility_check(
            "oblivious_large",
            feasibility_C_min=0.0,
            feasibility_C_max=100.0,
        )

        self.assertEqual(feasibility_check.keywords["C_min"], 0.0)
        self.assertEqual(feasibility_check.keywords["C_max"], 100.0)

    def test_generator_has_no_c_success_mode(self):
        parameters = inspect.signature(generate_heat_map_matrix).parameters

        self.assertNotIn("c_success_mode", parameters)

    @patch("heatmap_data_generation.heatmapDataGeneration.os.cpu_count")
    @patch("heatmap_data_generation.heatmapDataGeneration.smallest_D")
    @patch("heatmap_data_generation.heatmapDataGeneration.save_heat_map_matrix")
    @patch("heatmap_data_generation.heatmapDataGeneration.Pool")
    def test_worker_count_is_limited_by_num_samples(
        self,
        mock_pool,
        mock_save,
        mock_smallest_D,
        mock_cpu_count,
    ):
        mock_cpu_count.return_value = 32
        mock_pool.return_value.starmap.return_value = [
            (np.array([1.0]), 0.8),
            (np.array([0.0]), 0.8),
        ]
        mock_smallest_D.return_value = {"smallest_D": 2}

        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path.cwd()
            try:
                temporary_path = Path(temporary_directory)
                (temporary_path / "q_e").mkdir()
                os.chdir(temporary_path)
                generate_heat_map_matrix(
                    D_vs_TYPE="D_vs_T",
                    D_sample_sizes=np.array([1]),
                    num_samples=2,
                    T_max=1,
                    x=np.ones(1),
                    q=0.8,
                    n=1,
                    c=0.1,
                    corruption_type="adversarial",
                    beta=0.1,
                    random_seed=123,
                )
            finally:
                os.chdir(working_directory)

        mock_pool.assert_called_once_with(processes=2)
        task_arguments = mock_pool.return_value.starmap.call_args.args[1]
        self.assertNotEqual(task_arguments[0][-1], task_arguments[1][-1])
        mock_pool.return_value.close.assert_called_once_with()
        mock_pool.return_value.join.assert_called_once_with()

    @patch(
        "heatmap_data_generation.heatmapDataGeneration."
        "streaming_subsampled_qRK_step"
    )
    def test_D_vs_beta_uses_preset_c(self, mock_step):
        mock_step.return_value = (np.array([1.0 - np.sqrt(0.5)]), 0.8)
        common = dict(
            D=1,
            T_max=1,
            x=np.ones(1),
            q=0.8,
            beta=0.1,
            n=1,
            corruption_type="adversarial",
            c_min=0.0,
            c_max=1.0,
            s_min=0.0,
            s_max=1.0,
        )

        self.assertTrue(run_qRK_subsample_D_vs_beta(c=0.4, **common))
        self.assertFalse(run_qRK_subsample_D_vs_beta(c=0.6, **common))

    @patch(
        "heatmap_data_generation.heatmapDataGeneration."
        "streaming_subsampled_qRK_step"
    )
    def test_D_vs_T_uses_preset_c_at_each_time(self, mock_step):
        mock_step.return_value = (np.array([1.0 - np.sqrt(0.5)]), 0.8)
        common = dict(
            D=1,
            T_max=2,
            T_intervals=1,
            x=np.ones(1),
            q=0.8,
            beta=0.1,
            n=1,
            corruption_type="adversarial",
            c_min=0.0,
            c_max=1.0,
            s_min=0.0,
            s_max=1.0,
        )

        successes, _ = run_qRK_subsample_D_vs_T(c=0.4, **common)

        np.testing.assert_array_equal(successes, np.array([1.0, 0.0]))

    def test_saved_filename_has_no_bound_suffix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path.cwd()
            try:
                temporary_path = Path(temporary_directory)
                (temporary_path / "heat_map_raw_data").mkdir()
                os.chdir(temporary_path)
                save_heat_map_matrix(
                    D_vs_TYPE="D_vs_T",
                    data_type="",
                    mean_success=np.array([[1.0]]),
                    n=1,
                    D_sample_sizes=np.array([1]),
                    num_samples=1,
                    T_max=1,
                    q=0.8,
                    c=0.1,
                    corruption_type="adversarial",
                    beta=0.1,
                    T_intervals=1,
                )
                filenames = [path.name for path in (temporary_path / "heat_map_raw_data").iterdir()]
            finally:
                os.chdir(working_directory)

        self.assertEqual(len(filenames), 1)
        self.assertNotIn("c_bound", filenames[0])
        self.assertNotIn("c_success", filenames[0])

    @patch("heatmap_data_generation.heatmapDataGeneration.smallest_D")
    @patch("heatmap_data_generation.heatmapDataGeneration.save_heat_map_matrix")
    @patch("heatmap_data_generation.heatmapDataGeneration.Pool")
    def test_small_D_vs_T_run_has_expected_shapes(
        self,
        mock_pool,
        mock_save,
        mock_smallest_D,
    ):
        mock_pool.return_value.starmap.return_value = [
            (np.array([1.0, 0.0]), 0.8)
        ]
        mock_smallest_D.return_value = {"smallest_D": 2}

        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path.cwd()
            try:
                temporary_path = Path(temporary_directory)
                (temporary_path / "q_e").mkdir()
                os.chdir(temporary_path)
                generate_heat_map_matrix(
                    D_vs_TYPE="D_vs_T",
                    D_sample_sizes=np.array([1, 2]),
                    num_samples=1,
                    T_max=2,
                    x=np.ones(1),
                    q=0.8,
                    n=1,
                    c=0.1,
                    corruption_type="adversarial",
                    beta=0.1,
                    T_intervals=1,
                )
            finally:
                os.chdir(working_directory)

        saved = {call.kwargs["data_type"]: call.kwargs for call in mock_save.call_args_list}
        self.assertEqual(np.asarray(saved[""]["mean_success"]).shape, (2, 2))
        self.assertEqual(np.asarray(saved["D_min"]["mean_success"]).shape, (2, 1))
        self.assertNotIn("c_success_mode", saved[""])

    @patch("heatmap_data_generation.heatmapDataGeneration.smallest_D")
    @patch("heatmap_data_generation.heatmapDataGeneration.save_heat_map_matrix")
    @patch("heatmap_data_generation.heatmapDataGeneration.Pool")
    def test_small_D_vs_beta_run_has_expected_shape(
        self,
        mock_pool,
        mock_save,
        mock_smallest_D,
    ):
        mock_pool.return_value.starmap.return_value = [True]
        mock_smallest_D.return_value = {"smallest_D": 2}

        generate_heat_map_matrix(
            D_vs_TYPE="D_vs_beta",
            D_sample_sizes=np.array([1, 2]),
            num_samples=1,
            T_max=1,
            x=np.ones(1),
            q=0.8,
            n=1,
            c=0.1,
            corruption_type="adversarial",
            beta_samples=np.array([0.01, 0.02]),
        )

        success_call = next(
            call for call in mock_save.call_args_list
            if call.kwargs["data_type"] == ""
        )
        self.assertEqual(np.asarray(success_call.kwargs["mean_success"]).shape, (2, 2))
        self.assertNotIn("c_success_mode", success_call.kwargs)


if __name__ == "__main__":
    unittest.main()
