#!/usr/bin/env bash
set -euo pipefail
cd /root/fbhf_trackB
echo "[START] $(date)" > run_s05.status
/root/miniconda3/bin/python -u trackB_train.py --train --size 96 --K 40 --batch 16 --iters 3000 --eval_every 100 --sigma 0.5 --ckpt best_s05.pt > train_s05.log 2>&1
echo "[TRAIN_DONE] $(date)" >> run_s05.status
/root/miniconda3/bin/python -u trackB_train.py --eval --size 96 --K_eval 400 --ntest 8 --sigma 0.5 --ckpt best_s05.pt > eval_s05.log 2>&1
cp -f trackB_eval.png trackB_eval_s05.png || true
echo "[EVAL_DONE] $(date)" >> run_s05.status
