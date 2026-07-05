#!/usr/bin/env bash
set -euo pipefail
cd /root/fbhf_trackB_more/s09
echo "[START] $(date)" > run.status
/root/miniconda3/bin/python -u trackB_train.py --train --size 96 --K 40 --batch 16 --iters 3000 --eval_every 100 --sigma 0.9 --ckpt best_s09.pt > train_s09.log 2>&1
echo "[TRAIN_DONE] $(date)" >> run.status
/root/miniconda3/bin/python -u trackB_train.py --eval --size 96 --K_eval 400 --ntest 8 --sigma 0.9 --ckpt best_s09.pt > eval_s09.log 2>&1
cp -f trackB_eval.png trackB_eval_s09.png || true
echo "[EVAL_DONE] $(date)" >> run.status
