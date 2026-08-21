"""
Error-increase functions for adversarial (worst-case) Massart noise.

In the adversarial setting the corruption value epsilon can depend
arbitrarily on the row a_{r_{k+1}} and the quantile Q_{q,k+1}.  The
worst-case bound is computed by assuming the adversary always chooses the
sign of epsilon to maximise the update error.
"""

import numpy as np
from ..core.quantile import integrate_gaussian


def error_increased(qq: float) -> float:
    """One-step relative error increase for adversarial positive noise.

    Computes E[(Z + qq)^2] - E[Z^2] integrated against the standard normal
    density over the piecewise region that accounts for the threshold qq.
    The result is the net increase (relative to baseline E[Z^2] = 1).

    Parameters
    ----------
    qq : float
        Quantile threshold (half-normal quantile corresponding to 1 - alpha').

    Returns
    -------
    float
        Net error increase (>= 0 when qq > 0).
    """
    return (
        integrate_gaussian(lambda z: (z + qq) ** 2, 0, np.inf)
        + integrate_gaussian(lambda z: z ** 2, -np.inf, -qq / 2)
        + integrate_gaussian(lambda z: (z + qq) ** 2, -qq / 2, 0)
        - 1
    )
