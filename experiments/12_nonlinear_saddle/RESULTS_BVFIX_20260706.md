# Runtime-tested nonlinear saddle results after BV fix

This note records the final manuscript-facing nonlinear-B diagnostic after the
runtime-test bounded-variation bug was fixed.

## Certification rule

Shrink and fallback candidates are passed through the same bounded-variation
limiter as the original metric proposal. A fallback candidate is no longer
accepted automatically: it must pass the local runtime test. The JSON diagnostic
field `unresolved` counts any accepted endpoint that still fails the local test.
Only runs with `unresolved = 0` are used as certified results.

## Discarded easy run

Directory:
`nonlinear_tested_fixed_ext20_bvfix_c10_20260706b`

The aggressive fixed run reaches the numerical floor, but it is not certified
after the BV fix:

- fixed tested error: `7.8269e-06`
- charged B calls: `492.84375`
- min margin: `-544.7574`
- unresolved: `203`

This run is retained as a diagnostic artifact only.

## Manuscript run

Directory:
`nonlinear_hard_lam001_fixed_bvfix_20260706b`

Settings:

- `lambda = rho = 0.01`
- `eval_steps = 300`
- `ref_steps = 50000`
- fixed grid: `{3.5, 5, 7, 10, 14, 20, 30} chi` for both metric blocks
- `bv_c = 10`

Main results:

| method | step profile | mean rel. error | charged B calls | unresolved |
|---|---|---:|---:|---:|
| plain FBHF | `0.9 chi` | `7.9274e-01` | `600` | `0` |
| fixed tested warped FBHF | `(30 chi, 7 chi)` | `1.8832e-03` | `600` | `0` |
| learned tested warped FBHF | learned near `(30 chi, 7 chi)` | `1.8864e-03` | `600.125` | `0` |

Diagnostics:

- fixed tested: max local ratio `0.6458`, min margin `0.3442`, no backtracks;
- learned tested: max local ratio `0.9433`, min margin `0.0467`, four backtracks;
- endpoint learning gain is not present; the learned proposal matches the
  tested fixed metric within numerical noise.

Conclusion:

The runtime test removes the artificial global-chi parameter bottleneck and
certifies much larger local steps on this nonlinear monotone-Lipschitz
diagnostic. Once that bottleneck is removed, a tuned fixed metric already
captures the available geometry in this benchmark.
