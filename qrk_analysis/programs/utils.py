"""
Shared utilities for the programs sub-package.
"""

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – no GUI window, no pause
import matplotlib.pyplot as plt

# Root of the figures output tree, relative to this file's directory.
_FIGURES_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def figures_dir(*sub: str) -> str:
    """Return (and create if needed) a sub-directory under the figures root.

    Parameters
    ----------
    *sub : str
        Optional path components appended after the figures root.
        E.g. figures_dir("adversarial") -> .../figures/adversarial/

    Returns
    -------
    str
        Absolute path to the directory.
    """
    path = os.path.abspath(os.path.join(_FIGURES_ROOT, *sub))
    os.makedirs(path, exist_ok=True)
    return path


def savefig(filename: str, subdir: str = "", dpi: int = 150) -> str:
    """Save the current matplotlib figure and then show it.

    Parameters
    ----------
    filename : str
        File name including extension, e.g. "beta_vs_q_adversarial_orig.pdf".
    subdir : str
        Sub-folder inside figures/, e.g. "adversarial".
    dpi : int
        Resolution for raster formats.

    Returns
    -------
    str
        Full path where the figure was saved.
    """
    path = os.path.join(figures_dir(subdir), filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")
    return path
