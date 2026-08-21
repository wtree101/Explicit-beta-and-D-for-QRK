import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import convergence_curves_D_demo


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HeatmapDemoWiringTests(unittest.TestCase):
    @patch("experiments.heatmaps.generate_heat_map_matrix")
    def test_D_vs_T_demo_calls_public_heatmap_api(self, mock_generate):
        runpy.run_path(
            PROJECT_ROOT / "heatmap_generation_D_vs_T_demo.py",
            run_name="__main__",
        )

        mock_generate.assert_called_once()
        self.assertEqual(mock_generate.call_args.kwargs["D_vs_TYPE"], "D_vs_T")

    @patch("experiments.heatmaps.generate_heat_map_matrix")
    def test_D_vs_beta_demo_calls_public_heatmap_api(self, mock_generate):
        runpy.run_path(
            PROJECT_ROOT / "heatmap_generation_D_vs_beta_demo.py",
            run_name="__main__",
        )

        mock_generate.assert_called_once()
        self.assertEqual(mock_generate.call_args.kwargs["D_vs_TYPE"], "D_vs_beta")

    def test_convergence_demo_main_uses_public_simulation(self):
        config = convergence_curves_D_demo.ExperimentConfig()
        x = np.ones(config.n)
        errors = np.zeros(
            (
                len(config.D_list),
                config.num_trials,
                config.T // config.record_every + 1,
            )
        )
        figure = Mock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path.cwd()
            try:
                os.chdir(temporary_directory)
                with (
                    patch.object(
                        convergence_curves_D_demo,
                        "run_experiment",
                        return_value=(x, errors),
                    ) as mock_run,
                    patch.object(
                        convergence_curves_D_demo,
                        "plot_results",
                        return_value=figure,
                    ) as mock_plot,
                    patch.object(
                        convergence_curves_D_demo.np,
                        "savez",
                    ) as mock_save,
                ):
                    convergence_curves_D_demo.main()
            finally:
                os.chdir(working_directory)

        mock_run.assert_called_once()
        mock_plot.assert_called_once_with(mock_run.call_args.args[0], errors)
        figure.savefig.assert_called_once()
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
