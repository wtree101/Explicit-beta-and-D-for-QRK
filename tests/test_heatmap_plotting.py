import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heatmap_data_display.plot_heatmaps import (
    ColorMapping,
    PathOverrides,
    create_figure,
    load_heatmap_data,
    main,
)
from heatmap_data_display.profiles import HeatmapProfile


class HeatmapPlottingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        self.profile = HeatmapProfile(
            name="fixture",
            kind="D_vs_beta",
            model="Fixture",
            success_file="success.txt",
            boundary_file="boundary.txt",
            x_grid_file="x.txt",
            d_grid_file="d.txt",
            preview_file="fixture.pdf",
            paper_file="fixture.pdf",
        )
        np.savetxt(self.data_dir / "x.txt", [0.0, 0.01, 0.02])
        np.savetxt(self.data_dir / "d.txt", [2.0, 4.0])
        np.savetxt(self.data_dir / "boundary.txt", [2.5, 3.0, 3.5])
        np.savetxt(
            self.data_dir / "success.txt",
            [[0.1, 0.2, 0.3], [0.8, 0.9, 1.0]],
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_expected_D_by_horizontal_shape(self):
        data = load_heatmap_data(self.profile, data_dir=self.data_dir)

        self.assertEqual(data.success.shape, (2, 3))
        np.testing.assert_allclose(data.boundary, [2.5, 3.0, 3.5])

    def test_transposes_only_when_reverse_shape_matches(self):
        success = np.loadtxt(self.data_dir / "success.txt")
        np.savetxt(self.data_dir / "success.txt", success.T)

        data = load_heatmap_data(self.profile, data_dir=self.data_dir)

        self.assertEqual(data.success.shape, (2, 3))
        np.testing.assert_allclose(data.success, success)

    def test_rejects_invalid_cached_data(self):
        cases = {
            "non-finite success": np.array([[0.1, np.nan, 0.3], [0.8, 0.9, 1.0]]),
            "out-of-range success": np.array([[0.1, 1.1, 0.3], [0.8, 0.9, 1.0]]),
            "wrong shape": np.ones((4, 4)),
        }
        for label, values in cases.items():
            with self.subTest(label=label):
                np.savetxt(self.data_dir / "success.txt", values)
                with self.assertRaises(ValueError):
                    load_heatmap_data(self.profile, data_dir=self.data_dir)

    def test_rejects_boundary_length_mismatch(self):
        np.savetxt(self.data_dir / "boundary.txt", [2.5, 3.0])

        with self.assertRaisesRegex(ValueError, "boundary has length"):
            load_heatmap_data(self.profile, data_dir=self.data_dir)

    def test_missing_file_error_names_the_input(self):
        (self.data_dir / "success.txt").unlink()

        with self.assertRaisesRegex(FileNotFoundError, "success matrix"):
            load_heatmap_data(self.profile, data_dir=self.data_dir)

    def test_cli_lists_explicit_profiles(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["--list-profiles"])

        self.assertEqual(exit_code, 0)
        self.assertIn("d-vs-t-massart", output.getvalue())
        self.assertIn("d-vs-t-oblivious", output.getvalue())
        self.assertIn("d-vs-beta-massart", output.getvalue())
        self.assertIn("d-vs-beta-oblivious", output.getvalue())

    def test_cli_path_overrides_create_preview_pdf(self):
        output_path = self.data_dir / "preview.pdf"
        arguments = [
            "--profile",
            "d-vs-beta-massart",
            "--success",
            str(self.data_dir / "success.txt"),
            "--boundary",
            str(self.data_dir / "boundary.txt"),
            "--x-grid",
            str(self.data_dir / "x.txt"),
            "--d-grid",
            str(self.data_dir / "d.txt"),
            "--output",
            str(output_path),
            "--d-min",
            "2.25",
            "--d-max",
            "3.75",
            "--color-scale",
            "threshold",
            "--color-center",
            "0.9",
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = main(arguments)

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_path.is_file())
        self.assertGreater(output_path.stat().st_size, 1_000)

    def test_color_mappings_emphasize_values_near_one(self):
        linear = ColorMapping().normalization()
        threshold = ColorMapping(scale="threshold", center=0.9).normalization()
        power = ColorMapping(scale="power", gamma=2.0).normalization()

        self.assertAlmostEqual(float(linear(0.9)), 0.9)
        self.assertAlmostEqual(float(threshold(0.9)), 0.5)
        self.assertAlmostEqual(float(threshold(0.95)), 0.75)
        self.assertAlmostEqual(float(power(0.9)), 0.81)

    def test_rejects_invalid_nonlinear_color_parameters(self):
        with self.assertRaisesRegex(ValueError, "center"):
            ColorMapping(scale="threshold", center=1.0).normalization()
        with self.assertRaisesRegex(ValueError, "gamma"):
            ColorMapping(scale="power", gamma=0.0).normalization()

    def test_D_limits_default_to_data_and_allow_one_sided_override(self):
        data = load_heatmap_data(self.profile, data_dir=self.data_dir)
        default_figure = create_figure(self.profile, data)
        limited_figure = create_figure(self.profile, data, d_max=3.5)
        try:
            self.assertEqual(default_figure.axes[0].get_ylim(), (1.0, 5.0))
            self.assertEqual(limited_figure.axes[0].get_ylim(), (1.0, 3.5))
        finally:
            default_figure.clear()
            limited_figure.clear()

    def test_rejects_nonoverlapping_D_limits(self):
        data = load_heatmap_data(self.profile, data_dir=self.data_dir)

        with self.assertRaisesRegex(ValueError, "does not overlap"):
            create_figure(self.profile, data, d_min=10.0, d_max=20.0)

    def test_cli_does_not_allow_paper_output_override(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            main(
                [
                    "--profile",
                    "d-vs-t-massart",
                    "--paper",
                    "--output",
                    str(self.data_dir / "unexpected.pdf"),
                ]
            )

    def test_path_overrides_are_explicit(self):
        replacement = self.data_dir / "replacement.txt"
        np.savetxt(replacement, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        data = load_heatmap_data(
            self.profile,
            data_dir=self.data_dir,
            overrides=PathOverrides(success=replacement),
        )

        np.testing.assert_allclose(data.success, 1.0)


if __name__ == "__main__":
    unittest.main()
