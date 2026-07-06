# Nonlinear saddle benchmark

This folder implements the optional nonlinear-B benchmark used to test the
problem class where FBHF is structurally better motivated than primal-dual TV
splitting methods.

Problem family:

```text
min_x max_{y in Delta_m} sum_i y_i h_delta(a_i^T x - b_i)
    + lambda/2 ||x||^2 - rho/2 ||y - 1/m||^2.
```

The monotone inclusion uses

- `A = (0, N_Delta)`, so the only prox is simplex projection;
- `C(x,y) = (lambda x, rho(y - 1/m))`, which is cocoercive;
- `B(x,y) = (sum_i y_i grad h_i(x), -h(Ax-b))`, a nonlinear saddle
  coupling. It is monotone and Lipschitz on the Huber model, but it is not the
  skew linear coupling used in the TV experiment.

Main command:

```bash
python nonlinear_saddle_experiment.py --outdir nonlinear_run_20260706
```

Quick smoke test:

```bash
python nonlinear_saddle_experiment.py \
  --outdir smoke_local --n 40 --m 80 --batch 4 --ntest 4 \
  --train_iters 2 --eval_every 1 --train_K 5 --eval_steps 20 --ref_steps 80 \
  --skip_scan
```

The learned warped-metric pilot uses coordinatewise primal steps and
samplewise dual steps. The dual-step network receives per-sample Huber values,
dual variables, gradients, and one-step history features, so the benchmark can
test whether outlier-level structure gives learning a fair advantage over the
best fixed warped metric. When samplewise dual steps are used, the simplex
resolvent is the corresponding diagonal-metric weighted projection, not the
Euclidean simplex projection. The optional `--learn_anchor_tau_mult` and
`--learn_anchor_s_mult` flags initialise the learned metric around a tuned
fixed metric, which separates learning value from sigmoid saturation near a
step-size box boundary.

Runtime-tested certification is available with `--runtime_test`. In this mode,
each proposed warped step is accepted only when the local metric-adapted test
passes on the realised pair `(z_k, p_k)`. Failed proposals are shrunk and
recomputed, and the extra `B` evaluations are added to the endpoint
`dominant_B_calls` field. Shrink and fallback candidates are passed through the
same bounded-variation limiter as the network proposal. The diagnostics report
`metric_bv_sum`, `max_metric_jump`, and `unresolved`; only runs with
`unresolved = 0` should be read as certified.

The harder diagnostic used in the manuscript lowers the strong monotonicity
parameters and expands the tested grid:

```bash
python nonlinear_saddle_experiment.py \
  --outdir nonlinear_hard_lam001_fixed_bvfix \
  --runtime_test --bv_c 10 --lam 0.01 --rho 0.01 \
  --eval_steps 300 --ref_steps 50000 \
  --tlo 0.25 --thi 30.0 --slo 0.25 --shi 30.0 \
  --fixed_tau_mult 3.5 5.0 7.0 10.0 14.0 20.0 30.0 \
  --fixed_s_mult 3.5 5.0 7.0 10.0 14.0 20.0 30.0
```

Outputs include CSV/JSON metadata and publication-style PDF/PNG/SVG figures:

- `main/main_summary.csv`
- `main/main_curves.csv`
- `main/main_diagnostics.json`
- `beta_scan/beta_scan_summary.csv`
- `selection/selection_summary.csv`
