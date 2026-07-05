#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/root/miniconda3/bin/python
SEED=11
TEST_SEED=20240704
COMMON="--size 96 --K 40 --iters 3000 --eval_every 100 --batch 16"
EVAL_COMMON="--size 96 --K_eval 400 --ntest 16 --test_seed ${TEST_SEED}"
{
  echo "[full-run] start $(date)"
  echo "[full-run] selftest"
  $PY -u trackB_warped_train.py --selftest --seed ${SEED} 2>&1 | tee selftest.log
  echo "[full-run] train bv"
  $PY -u trackB_warped_train.py --train --mode bv --seed ${SEED} ${COMMON} --ckpt wbv_seed11.pt 2>&1 | tee train_bv.log
  echo "[full-run] eval bv"
  $PY -u trackB_warped_train.py --eval --mode bv --seed ${SEED} ${EVAL_COMMON} --ckpt wbv_seed11.pt --curves_csv curves_bv.csv --metrics_json metrics_bv.json --plot_png eval_bv_quick.png 2>&1 | tee eval_bv.log
  cp -f eval_bv_quick.png eval_bv_raw.png
  echo "[full-run] train free"
  $PY -u trackB_warped_train.py --train --mode free --seed ${SEED} ${COMMON} --ckpt wfree_seed11.pt 2>&1 | tee train_free.log
  echo "[full-run] eval free"
  $PY -u trackB_warped_train.py --eval --mode free --seed ${SEED} ${EVAL_COMMON} --ckpt wfree_seed11.pt --curves_csv curves_free.csv --metrics_json metrics_free.json --plot_png eval_free_quick.png 2>&1 | tee eval_free.log
  cp -f eval_free_quick.png eval_free_raw.png
  echo "[full-run] done $(date)"
} > full_run.log 2>&1
