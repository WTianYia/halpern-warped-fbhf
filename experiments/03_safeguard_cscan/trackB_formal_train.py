import argparse
import math
from pathlib import Path

import torch

from trackB_formal_eval import DEV, PrecNet, TV, make_batch, normD, set_seed, unroll


def build_cfg(args, L):
    return dict(
        tlo=0.1,
        thi=1.8,
        slo=0.02,
        shi=0.12,
        rho=0.9,
        tau0=0.3,
        s0=0.1,
        c=args.c,
    )


def objective_after(prob, xs):
    weights = torch.arange(1, len(xs) + 1, device=xs[0].device, dtype=xs[0].dtype)
    vals = torch.stack([prob.obj(x).mean() for x in xs])
    return (weights * vals).sum() / weights.sum()


@torch.no_grad()
def val_objective(net, args, L, cfg, seed):
    b, otf, mu, _ = make_batch(
        args.batch,
        args.size,
        args.size,
        DEV,
        seed,
        args.data,
        args.blur,
        args.noise,
        args.image_dir,
    )
    prob = TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    learned = prob.obj(unroll(prob, x0, y0, args.K, net=net, mode=args.mode, L=L, cfg=cfg)[-1]).mean().item()
    fixed = prob.obj(unroll(prob, x0, y0, args.K, L=L, cfg=cfg, fixed=(1.5, 0.07))[-1]).mean().item()
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))
    plain = prob.obj(unroll(prob, x0, y0, args.K, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))[-1]).mean().item()
    return learned, fixed, plain


def train(args):
    set_seed(args.seed)
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    L = normD(args.size, args.size, DEV)
    cfg = build_cfg(args, L)
    net = PrecNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    best = float("inf")
    for it in range(1, args.iters + 1):
        b, otf, mu, _ = make_batch(
            args.batch,
            args.size,
            args.size,
            DEV,
            args.seed * 100000 + it,
            args.data,
            args.blur,
            args.noise,
            args.image_dir,
        )
        prob = TV(b, otf, mu)
        x0 = b.clone()
        y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
        xs = unroll(prob, x0, y0, args.K, net=net, mode=args.mode, L=L, cfg=cfg)
        loss = objective_after(prob, xs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % args.eval_every == 0 or it == 1:
            net.eval()
            learned, fixed, plain = val_objective(net, args, L, cfg, args.val_seed)
            net.train()
            tag = ""
            if math.isfinite(learned) and learned < best:
                best = learned
                torch.save(net.state_dict(), args.ckpt)
                tag = " *saved"
            print(
                f"it{it:5d} loss {loss.item():.6e} | "
                f"val learned {learned:.6e} vs fixed {fixed:.6e} vs plain {plain:.6e} "
                f"(gain over fixed {fixed-learned:+.3e}){tag}",
                flush=True,
            )
    print(f"done best={best:.6e} -> {args.ckpt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image_dir", default="standard_images")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--val_seed", type=int, default=9011)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mode", choices=["bv", "free"], default="bv")
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--data", default="real")
    ap.add_argument("--blur", default="motion")
    ap.add_argument("--noise", default="high")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
