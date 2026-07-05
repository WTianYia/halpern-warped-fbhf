# Track B remote run result: sigma=0.5

Date: 2026-07-04

Remote GPU: NVIDIA GeForce RTX 3080 Ti, 12GB

Command:

```bash
/root/miniconda3/bin/python -u trackB_train.py \
  --train --size 96 --K 40 --batch 16 --iters 3000 \
  --eval_every 100 --sigma 0.5 --ckpt best_s05.pt

/root/miniconda3/bin/python -u trackB_train.py \
  --eval --size 96 --K_eval 400 --ntest 8 \
  --sigma 0.5 --ckpt best_s05.pt
```

Training:

- Best validation objective: 4.064
- Plain validation objective: 4.259
- Training completed on remote in about 17 minutes.

Held-out evaluation:

| Method | err@K/2 | err@K | PSNR@K |
|---|---:|---:|---:|
| plain | 7.07e-02 | 4.17e-02 | 28.01 |
| linesearch | 6.90e-02 | 3.91e-02 | 28.04 |
| inertial(alpha=0.2,gamma=0.228) | 7.39e-02 | 4.55e-02 | 27.98 |
| momentum-dev | 8.20e-02 | 7.26e-02 | 26.76 |
| learned-dev | 1.18e-02 | 6.79e-03 | 28.59 |

Verdict:

This run is a strong positive signal for Track B. The learned deviation beats both line-search FBHF and best-tuned inertial FBHF on fixed-budget relative primal error and PSNR.

Artifacts:

- `train_s05.log`
- `eval_s05.log`
- `best_s05.pt`
- `trackB_eval_s05.png`
- `run_s05.status`
- `run_s05.sh`
