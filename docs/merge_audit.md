# QRK analysis merge audit

## Baseline comparison

The pre-merge `qrk_analysis` and `qrk_adv` implementations were evaluated on
representative interior parameter values before compatibility forwarding was
introduced.

| Quantity | Maximum observed difference |
| --- | ---: |
| Bernoulli KL divergence | `0.0` |
| Truncated Gaussian second moment | `1.67e-16` |
| Fixed-noise error increase | `0.0` |
| Adversarial contraction | `8.33e-17` |
| Fixed-supremum contraction | `1.11e-16` |
| Gaussian-supremum contraction | `1.11e-16` |
| Gaussian batch expectation vs scalar quadrature | below `1e-8` |

No formula-level conflict was found. Both revised oblivious checks use the
paper interval

```text
[alpha / (1 - beta), 1 - alpha_prime / (1 - beta)]
```

and use the same scaled lower quantile in the lower-event penalty.

## Search discrepancy

The old `qrk_analysis` search constrained `alpha_prime` to a grid beginning at
`0.001`. The old `qrk_adv` search instead bisected the failure-probability
constraint. This changes numerical bounds without changing the theorem.

For `q=0.75`, `beta=0.01`, `T=20000`, `delta_f=0.1`, and `c=0`:

- Massart: both searches certify integer `D=25`.
- Fixed-noise supremum with the paper grids: the old grid certifies `D=14`;
  the strict-interior bisection certifies `D=13`.

The canonical implementation uses strict-interior bisection and integer search
in `D`. Paper figures and prose are regenerated from this implementation.

## Compatibility removal

The temporary `qrk_adv` forwarding package was removed after all maintained
callers migrated to direct `qrk_analysis` imports. The historical comparison
above retains the old package names to document the pre-merge audit.

The temporary `heatmap_data_generation` facade was likewise removed after the
heatmap drivers and tests migrated to the canonical `experiments.heatmaps`
API. Neither numerical theory nor simulation now has a top-level forwarding
package.
