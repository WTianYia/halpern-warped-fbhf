# Track B warped-preconditioned FBHF experiment

Date: 2026-07-04

Remote run folder: `/root/fbhf_trackB_warped_20260704_201955`

Local snapshot:
`C:\Users\33084\Desktop\唐玉超老师的文章\06_锚定FBHF定稿方案\trackB_训练实验\remote_warped_20260704_2055`

## Configuration

- Script: `trackB_warped_train.py`
- Problem: TV deblurring primal-dual FBHF, image size 96
- Training: `K=40`, `iters=3000`, `batch=16`, `eval_every=100`
- Evaluation: `K_eval=400`, `ntest=8`
- Modes:
  - `bv`: bounded-variation metric update, theory-safe mode
  - `free`: free per-step metric update, empirical upper comparison

## Selftest

Remote selftest passed.

- K-adjoint error: `0.00e+00`
- D-adjoint error: `7.63e-06`
- `||D|| = 2.819`
- `chi ~= 0.3246`
- `1/chi ~= 3.080`
- fixed warped objective: `3.422 -> 1.909`

## Training Summary

| mode | best validation | fixed-precond validation | comment |
|---|---:|---:|---|
| `bv` | `4.008` | `4.500` | stable theory-safe gain |
| `free` | `3.980` | `4.495` | slightly better than `bv`, but not a large jump |

## Evaluation Summary

Median relative primal error.

| mode | method | err@K/2 | err@K |
|---|---|---:|---:|
| `bv` | plain-FBHF | `2.99e-02` | `6.24e-03` |
| `bv` | fixed-precond | `5.95e-03` | `3.56e-03` |
| `bv` | learned-warped | `4.52e-03` | `2.62e-03` |
| `free` | plain-FBHF | `3.05e-02` | `7.71e-03` |
| `free` | fixed-precond | `5.73e-03` | `3.70e-03` |
| `free` | learned-warped | `4.34e-03` | `2.53e-03` |

## Interpretation

The `bv` mode is green: the theory-safe learned warped preconditioner beats both plain FBHF and fixed preconditioning on the held-out evaluation.

The `free` mode is only marginally better than `bv`: `2.53e-03` vs `2.62e-03` at `K=400`. This means the bounded-variation constraint does not appear to be the main numerical bottleneck in this prototype. The useful gain comes from the warped/preconditioned structure itself, not from unconstrained per-step metric changes.

This supports using warped-preconditioned FBHF as the theory-safe learning route for Track B.

## Artifacts

- `trackB_warped_train.py`: source used on the remote host
- `train_bv.log`, `eval_bv.log`: bounded-variation training/evaluation logs
- `train_free.log`, `eval_free.log`: free-mode training/evaluation logs
- `wbv.pt`, `wfree.pt`: trained weights
- `trackB_warped_eval_bv.png`, `trackB_warped_eval_free.png`: evaluation curves
- `manifest.json`: downloaded file manifest
