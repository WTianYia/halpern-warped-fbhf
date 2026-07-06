# Halpern-Anchored and Learned Warped-Metric FBHF

This repository contains the experimental code and reproducibility assets for the manuscript

**Halpern-Anchored and Learned Warped-Metric Forward--Backward--Half-Forward Splitting for Three-Operator Monotone Inclusions**.

The experiments support two parts of the paper:

- Halpern-anchored FBHF for strong convergence and minimum-norm solution selection.
- Learned warped-metric FBHF for certified fixed-budget acceleration in TV image deblurring.

## Repository layout

```text
experiments/
  01_nonunique_selection/          # minimum-norm selection experiment
  02_main_oracle_real_synthetic/   # main real/synthetic oracle-budget benchmark
  03_safeguard_cscan/              # bounded-variation safeguard scan
  04_distribution_shift_ood/       # noise, blur, and motion-blur shift tests
  05_reconstruction_plate/         # image restoration visual comparison
  06_warped_training_runs/         # warped-metric training logs, metrics, checkpoints
  07_supporting_probes/            # negative probes used to rule out weaker learning routes
  08_revision_experiments/         # reviewer-requested operator-cap diagnostics
  09_m2_m3_supplement/             # added baseline and data-protocol experiments
  10_certified_mode_supplement/    # fully certified operating-box supplement
  11_high_precision_reference/     # high-precision reference audits
  12_nonlinear_saddle/             # nonlinear-B saddle benchmark scaffold
```

Each experiment folder includes the relevant source code, raw CSV/JSON data, generated figures, and metadata needed to reproduce or audit the reported results.

## Main scripts

- `experiments/02_main_oracle_real_synthetic/trackB_formal_train.py`
- `experiments/02_main_oracle_real_synthetic/trackB_formal_eval.py`
- `experiments/02_main_oracle_real_synthetic/make_oracle_std_figures.py`
- `experiments/01_nonunique_selection/run_nonunique_selection.py`
- `experiments/06_warped_training_runs/warped_main/trackB_warped_train.py`
- `experiments/12_nonlinear_saddle/nonlinear_saddle_experiment.py`

## Notes

The repository contains both final experiments and supporting probes. The supporting probes are included for transparency: they document why scalar step learning and free additive deviations were not used as the main certified learning mechanism.

The manuscript uses matched oracle-budget comparisons. Wall-clock timing is not used as a headline metric because GPU network evaluation and CPU/GPU synchronization would make the comparison hardware-dependent.

## Environment

The training scripts use Python with PyTorch, NumPy, SciPy, scikit-image, pandas, and matplotlib. GPU acceleration is recommended for training the learned warped metric, while most evaluation and plotting scripts can run on CPU.

## Citation

The final bibliographic information will be added after publication.
