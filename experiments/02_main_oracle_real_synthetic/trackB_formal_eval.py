import argparse
import csv
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_URLS = {
    "camera": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.20.0/skimage/data/camera.png",
    "coffee": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.20.0/skimage/data/coffee.png",
    "astronaut": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.20.0/skimage/data/astronaut.png",
    "coins": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.20.0/skimage/data/coins.png",
    "page": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.20.0/skimage/data/page.png",
    "moon": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.20.0/skimage/data/moon.png",
    "chelsea": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.20.0/skimage/data/chelsea.png",
    "barbara": "https://raw.githubusercontent.com/semnan-university-ai/image-processing-benchmark/master/barbara.png",
    "boat": "https://raw.githubusercontent.com/semnan-university-ai/image-processing-benchmark/master/boat.png",
    "peppers": "https://raw.githubusercontent.com/semnan-university-ai/image-processing-benchmark/master/peppers.png",
    "cameraman_classic": "https://raw.githubusercontent.com/semnan-university-ai/image-processing-benchmark/master/cameraman.tif",
    "goldhill": "https://raw.githubusercontent.com/semnan-university-ai/image-processing-benchmark/master/goldhill.png",
    "baboon": "https://raw.githubusercontent.com/semnan-university-ai/image-processing-benchmark/master/baboon.png",
    "sails": "https://raw.githubusercontent.com/semnan-university-ai/image-processing-benchmark/master/sails.png",
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def gauss_otf(H, W, sigma, device):
    B = sigma.shape[0]
    yy = torch.arange(H, device=device).view(1, 1, H, 1) - H // 2
    xx = torch.arange(W, device=device).view(1, 1, 1, W) - W // 2
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma.view(B, 1, 1, 1) ** 2))
    g = g / g.sum(dim=(-2, -1), keepdim=True)
    return torch.fft.fft2(torch.fft.ifftshift(g, dim=(-2, -1)))


def motion_otf(H, W, length, device, horizontal=True):
    B = length.shape[0]
    ker = torch.zeros(B, 1, H, W, device=device)
    for i in range(B):
        L = int(length[i].item())
        L = max(3, min(L, min(H, W) // 2))
        if horizontal:
            ker[i, 0, H // 2, W // 2 - L // 2 : W // 2 - L // 2 + L] = 1.0 / L
        else:
            ker[i, 0, H // 2 - L // 2 : H // 2 - L // 2 + L, W // 2] = 1.0 / L
    return torch.fft.fft2(torch.fft.ifftshift(ker, dim=(-2, -1)))


def Kf(x, otf):
    return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * otf))


def Ktf(x, otf):
    return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(otf)))


def Dop(x):
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.cat([gx, gy], dim=1)


def Dtop(p):
    px, py = p[:, 0:1], p[:, 1:2]
    ax = torch.zeros_like(px)
    ay = torch.zeros_like(py)
    ax[..., :, 0] = -px[..., :, 0]
    ax[..., :, 1:-1] = px[..., :, :-2] - px[..., :, 1:-1]
    ax[..., :, -1] = px[..., :, -2]
    ay[..., 0, :] = -py[..., 0, :]
    ay[..., 1:-1, :] = py[..., :-2, :] - py[..., 1:-1, :]
    ay[..., -1, :] = py[..., -2, :]
    return ax + ay


def projb(y, mu):
    n = torch.sqrt(y[:, 0:1] ** 2 + y[:, 1:2] ** 2 + 1e-12)
    return y * torch.clamp(mu / n, max=1.0)


class TV:
    def __init__(self, b, otf, mu):
        self.b, self.otf, self.mu = b, otf, mu

    def obj(self, x):
        d = Dop(x)
        tv = torch.sqrt(d[:, 0:1] ** 2 + d[:, 1:2] ** 2 + 1e-12).sum(dim=(1, 2, 3))
        return 0.5 * ((Kf(x, self.otf) - self.b) ** 2).sum(dim=(1, 2, 3)) + self.mu * tv


def normD(H, W, device):
    v = torch.randn(1, 1, H, W, device=device)
    for _ in range(60):
        v = Dtop(Dop(v))
        v = v / torch.norm(v)
    return torch.sqrt(torch.norm(Dtop(Dop(v))) / torch.norm(v)).item()


def warped_step(prob, x, y, tau, s):
    gx = Ktf(Kf(x, prob.otf) - prob.b, prob.otf) + Dtop(y)
    px = x - tau * gx
    py = projb(y + s * Dop(x), prob.mu)
    Tx = px + tau * Dtop(y - py)
    Ty = py - s * Dop(x - px)
    return (Tx, Ty), (px, py)


class PrecNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 2, 3, padding=1),
        )

    def forward(self, feat):
        return self.net(feat)


def clamp_ts(raw, L, tlo, thi, slo, shi, rho=0.9):
    tau = tlo + (thi - tlo) * torch.sigmoid(raw[:, 0:1])
    s = slo + (shi - slo) * torch.sigmoid(raw[:, 1:2])
    s = torch.minimum(s, rho / (tau * L**2 + 1e-9))
    return tau, s


def features(x, y, px, py, x_prev):
    return torch.cat([x, x - px, x - x_prev, y, y - py], dim=1)


def download_images(image_dir):
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    for name, url in IMAGE_URLS.items():
        out = image_dir / f"{name}.png"
        if not out.exists():
            try:
                urllib.request.urlretrieve(url, out)
            except Exception as exc:
                print(f"[image] skip {name}: {exc}")
                continue
        ok.append(out)
    return ok


def pil_to_gray_tensor(path, H, W):
    im = Image.open(path).convert("L").resize((W, H), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None]


def synthetic_piecewise(B, H, W, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    imgs = torch.zeros(B, 1, H, W)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    for i in range(B):
        for _ in range(int(torch.randint(4, 9, (1,), generator=g))):
            val = float(torch.rand(1, generator=g)) * 0.8 + 0.15
            kind = int(torch.randint(0, 3, (1,), generator=g))
            if kind == 0:
                x0 = int(torch.randint(0, W - 8, (1,), generator=g))
                y0 = int(torch.randint(0, H - 8, (1,), generator=g))
                w = int(torch.randint(8, max(9, W // 2), (1,), generator=g))
                h = int(torch.randint(8, max(9, H // 2), (1,), generator=g))
                imgs[i, 0, y0 : min(H, y0 + h), x0 : min(W, x0 + w)] = val
            elif kind == 1:
                cy = int(torch.randint(8, H - 8, (1,), generator=g))
                cx = int(torch.randint(8, W - 8, (1,), generator=g))
                r = int(torch.randint(4, max(5, H // 4), (1,), generator=g))
                imgs[i, 0][(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = val
            else:
                freq = float(torch.rand(1, generator=g) * 6 + 2)
                phase = float(torch.rand(1, generator=g) * 6.28)
                stripe = 0.5 + 0.5 * torch.sin((xx.float() / W + yy.float() / H) * freq * 6.28 + phase)
                imgs[i, 0] = torch.maximum(imgs[i, 0], val * stripe)
    return imgs.clamp(0, 1)


def image_batch(B, H, W, seed, data, image_dir):
    if data == "synthetic":
        return synthetic_piecewise(B, H, W, seed)
    files = download_images(image_dir)
    if not files:
        return synthetic_piecewise(B, H, W, seed)
    imgs = []
    rng = np.random.default_rng(seed)
    for i in range(B):
        p = files[int(rng.integers(0, len(files)))]
        x = pil_to_gray_tensor(p, H, W)
        # Random crop-like flips/intensity for more than 6 repeated instances.
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        gain = float(rng.uniform(0.85, 1.15))
        bias = float(rng.uniform(-0.05, 0.05))
        imgs.append((x * gain + bias).clamp(0, 1))
    return torch.cat(imgs, dim=0)


def make_batch(B, H, W, device, seed, data="synthetic", blur="train", noise="train", image_dir="images"):
    imgs = image_batch(B, H, W, seed, data, image_dir).to(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 17)
    if blur == "train":
        sigma = (torch.rand(B, generator=g) * 1.5 + 1.0).to(device)
        otf = gauss_otf(H, W, sigma, device)
    elif blur == "wide":
        sigma = (torch.rand(B, generator=g) * 2.0 + 2.0).to(device)
        otf = gauss_otf(H, W, sigma, device)
    elif blur == "motion":
        length = (torch.randint(7, 17, (B,), generator=g)).to(device)
        otf = motion_otf(H, W, length, device, horizontal=True)
    else:
        sigma = torch.full((B,), float(blur), device=device)
        otf = gauss_otf(H, W, sigma, device)
    if noise == "train":
        sig = (torch.rand(B, 1, 1, 1, generator=g) * 0.015 + 0.005).to(device)
    elif noise == "high":
        sig = torch.full((B, 1, 1, 1), 0.03, device=device)
    elif noise == "low":
        sig = torch.full((B, 1, 1, 1), 0.005, device=device)
    else:
        sig = torch.full((B, 1, 1, 1), float(noise), device=device)
    eps = torch.randn(B, 1, H, W, generator=g).to(device)
    b = (Kf(imgs, otf) + sig * eps).clamp(0, 1)
    return b, otf, 0.02, imgs


def unroll(prob, x0, y0, K, net=None, mode="bv", L=2.83, cfg=None, fixed=None, counters=None):
    x, y = x0, y0
    x_prev = x0
    xs = []
    tlo, thi, slo, shi = cfg["tlo"], cfg["thi"], cfg["slo"], cfg["shi"]
    if net is not None and mode == "bv":
        tau = torch.full_like(x, cfg["tau0"])
        s = torch.full_like(x, cfg["s0"])
    for k in range(K):
        if fixed is not None:
            tau = torch.full_like(x, fixed[0])
            s = torch.full_like(x, fixed[1])
        elif net is not None:
            _, (px0, py0) = warped_step(prob, x, y, torch.full_like(x, cfg["tau0"]), torch.full_like(x, cfg["s0"]))
            raw = net(features(x, y, px0, py0, x_prev))
            dtau, dsig = clamp_ts(raw, L, tlo, thi, slo, shi, cfg["rho"])
            if counters is not None:
                counters["network"] += 1
            if mode == "free":
                tau, s = dtau, dsig
            else:
                eta = cfg["c"] / ((k + 1) ** 1.1)
                tau = torch.clamp(tau + torch.clamp(dtau - tau, -eta, eta), tlo, thi)
                s = torch.clamp(s + torch.clamp(dsig - s, -eta, eta), slo, shi)
                s = torch.minimum(s, cfg["rho"] / (tau * L**2 + 1e-9))
        else:
            tau = torch.full_like(x, cfg["tau0"])
            s = torch.full_like(x, cfg["s0"])
        T, _ = warped_step(prob, x, y, tau, s)
        if counters is not None:
            counters["prox"] += 1
            counters["B"] += 2
            counters["C"] += 1
        x_prev = x
        x, y = T
        xs.append(x)
    return xs


def unroll_final(prob, x0, y0, K, net=None, mode="bv", L=2.83, cfg=None, fixed=None, counters=None):
    x, y = x0, y0
    x_prev = x0
    tlo, thi, slo, shi = cfg["tlo"], cfg["thi"], cfg["slo"], cfg["shi"]
    if net is not None and mode == "bv":
        tau = torch.full_like(x, cfg["tau0"])
        s = torch.full_like(x, cfg["s0"])
    for k in range(K):
        if fixed is not None:
            tau = torch.full_like(x, fixed[0])
            s = torch.full_like(x, fixed[1])
        elif net is not None:
            _, (px0, py0) = warped_step(prob, x, y, torch.full_like(x, cfg["tau0"]), torch.full_like(x, cfg["s0"]))
            raw = net(features(x, y, px0, py0, x_prev))
            dtau, dsig = clamp_ts(raw, L, tlo, thi, slo, shi, cfg["rho"])
            if counters is not None:
                counters["network"] += 1
            if mode == "free":
                tau, s = dtau, dsig
            else:
                eta = cfg["c"] / ((k + 1) ** 1.1)
                tau = torch.clamp(tau + torch.clamp(dtau - tau, -eta, eta), tlo, thi)
                s = torch.clamp(s + torch.clamp(dsig - s, -eta, eta), slo, shi)
                s = torch.minimum(s, cfg["rho"] / (tau * L**2 + 1e-9))
        else:
            tau = torch.full_like(x, cfg["tau0"])
            s = torch.full_like(x, cfg["s0"])
        T, _ = warped_step(prob, x, y, tau, s)
        if counters is not None:
            counters["prox"] += 1
            counters["B"] += 2
            counters["C"] += 1
        x_prev = x
        x, y = T
    return x


def rel_error(x, xstar, nx):
    return rel_error_vec(x, xstar, nx).mean().item()


def rel_error_vec(x, xstar, nx):
    return torch.sqrt(((x - xstar) ** 2).sum(dim=(1, 2, 3))) / nx


def unroll_trace(prob, x0, y0, K, xstar, nx, net=None, mode="bv", L=2.83, cfg=None, fixed=None, counters=None):
    x, y = x0, y0
    x_prev = x0
    errs = []
    tlo, thi, slo, shi = cfg["tlo"], cfg["thi"], cfg["slo"], cfg["shi"]
    if net is not None and mode == "bv":
        tau = torch.full_like(x, cfg["tau0"])
        s = torch.full_like(x, cfg["s0"])
    for k in range(K):
        if fixed is not None:
            tau = torch.full_like(x, fixed[0])
            s = torch.full_like(x, fixed[1])
        elif net is not None:
            _, (px0, py0) = warped_step(prob, x, y, torch.full_like(x, cfg["tau0"]), torch.full_like(x, cfg["s0"]))
            raw = net(features(x, y, px0, py0, x_prev))
            dtau, dsig = clamp_ts(raw, L, tlo, thi, slo, shi, cfg["rho"])
            if counters is not None:
                counters["network"] += 1
            if mode == "free":
                tau, s = dtau, dsig
            else:
                eta = cfg["c"] / ((k + 1) ** 1.1)
                tau = torch.clamp(tau + torch.clamp(dtau - tau, -eta, eta), tlo, thi)
                s = torch.clamp(s + torch.clamp(dsig - s, -eta, eta), slo, shi)
                s = torch.minimum(s, cfg["rho"] / (tau * L**2 + 1e-9))
        else:
            tau = torch.full_like(x, cfg["tau0"])
            s = torch.full_like(x, cfg["s0"])
        T, _ = warped_step(prob, x, y, tau, s)
        if counters is not None:
            counters["prox"] += 1
            counters["B"] += 2
            counters["C"] += 1
        x_prev = x
        x, y = T
        errs.append(rel_error(x, xstar, nx))
    return errs, rel_error_vec(x, xstar, nx).detach().cpu().numpy().tolist()


@torch.no_grad()
def unroll_linesearch(prob, x0, y0, K, chi, theta=0.9):
    x, y = x0, y0
    gamma = chi
    xs = []
    oracle_marks = []
    counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
    for _ in range(K):
        gf = Ktf(Kf(x, prob.otf) - prob.b, prob.otf)
        b0 = Dtop(y)
        b1 = -Dop(x)
        counters["C"] += 1
        counters["B"] += 1
        gamma = gamma * 1.2
        trials = 0
        for _trial in range(40):
            trials += 1
            ix = x - gamma * (gf + b0)
            iy = y - gamma * b1
            xr = ix
            yr = projb(iy, prob.mu)
            r0 = Dtop(yr)
            r1 = -Dop(xr)
            counters["prox"] += 1
            counters["B"] += 1
            lhs = gamma * torch.sqrt(((b0 - r0) ** 2).sum(dim=(1, 2, 3)) + ((b1 - r1) ** 2).sum(dim=(1, 2, 3)) + 1e-12)
            rhs = theta * torch.sqrt(((x - xr) ** 2).sum(dim=(1, 2, 3)) + ((y - yr) ** 2).sum(dim=(1, 2, 3)) + 1e-12)
            if bool((lhs <= rhs).all()) or gamma <= 1e-6:
                break
            gamma = gamma * 0.5
        counters["backtracks"] += max(0, trials - 1)
        x = xr + gamma * (b0 - r0)
        y = yr + gamma * (b1 - r1)
        xs.append(x)
        oracle_marks.append(counters["prox"] + counters["B"] + counters["C"])
    return xs, oracle_marks, counters


@torch.no_grad()
def unroll_linesearch_trace(prob, x0, y0, K, chi, xstar, nx, theta=0.9):
    x, y = x0, y0
    gamma = chi
    errs = []
    oracle_marks = []
    counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
    for _ in range(K):
        gf = Ktf(Kf(x, prob.otf) - prob.b, prob.otf)
        b0 = Dtop(y)
        b1 = -Dop(x)
        counters["C"] += 1
        counters["B"] += 1
        gamma = gamma * 1.2
        trials = 0
        for _trial in range(40):
            trials += 1
            ix = x - gamma * (gf + b0)
            iy = y - gamma * b1
            xr = ix
            yr = projb(iy, prob.mu)
            r0 = Dtop(yr)
            r1 = -Dop(xr)
            counters["prox"] += 1
            counters["B"] += 1
            lhs = gamma * torch.sqrt(((b0 - r0) ** 2).sum(dim=(1, 2, 3)) + ((b1 - r1) ** 2).sum(dim=(1, 2, 3)) + 1e-12)
            rhs = theta * torch.sqrt(((x - xr) ** 2).sum(dim=(1, 2, 3)) + ((y - yr) ** 2).sum(dim=(1, 2, 3)) + 1e-12)
            if bool((lhs <= rhs).all()) or gamma <= 1e-6:
                break
            gamma = gamma * 0.5
        counters["backtracks"] += max(0, trials - 1)
        x = xr + gamma * (b0 - r0)
        y = yr + gamma * (b1 - r1)
        errs.append(rel_error(x, xstar, nx))
        oracle_marks.append(counters["prox"] + counters["B"] + counters["C"])
    return errs, rel_error_vec(x, xstar, nx).detach().cpu().numpy().tolist(), oracle_marks, counters


@torch.no_grad()
def evaluate_one(args, data, blur, noise, c, mode):
    set_seed(args.seed)
    H = W = args.size
    L = normD(H, W, DEV)
    cfg = dict(tlo=0.1, thi=1.8, slo=0.02, shi=0.12, rho=0.9, tau0=0.3, s0=0.1, c=c)
    b, otf, mu, clean = make_batch(args.ntest, H, W, DEV, args.test_seed, data, blur, noise, args.image_dir)
    prob = TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))
    xstar = unroll_final(prob, x0, y0, args.ref_iters, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))
    nx = torch.sqrt((xstar**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)

    net = None
    if mode in ("bv", "free"):
        net = PrecNet().to(DEV)
        net.load_state_dict(torch.load(args.ckpt, map_location=DEV))
        net.eval()

    specs = {
        "plain-FBHF": dict(fixed=(chi - 0.05 * chi, 0.10), net=None, mode="bv"),
        "line-search-FBHF": dict(fixed="linesearch", net=None, mode="bv"),
        "fixed-precond": dict(fixed=(1.5, 0.07), net=None, mode="bv"),
        f"learned-warped-{mode}-c{c:g}": dict(fixed=None, net=net, mode=mode),
    }
    rows = []
    summary = []
    per_image_rows = []
    for method, spec in specs.items():
        counters = {"prox": 0, "B": 0, "C": 0, "network": 0, "backtracks": 0}
        # Warm up CUDA kernels / network forward. This is excluded from wall-clock curves.
        warm_k = min(3, args.K_eval)
        if spec["fixed"] == "linesearch":
            _ = unroll_linesearch(prob, x0, y0, warm_k, chi=chi)
        else:
            _ = unroll(prob, x0, y0, warm_k, net=spec["net"], mode=spec["mode"], L=L, cfg=cfg, fixed=spec["fixed"])
        if DEV == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        oracle_marks = None
        if spec["fixed"] == "linesearch":
            errs, endpoint_vec, oracle_marks, counters = unroll_linesearch_trace(prob, x0, y0, args.K_eval, chi=chi, xstar=xstar, nx=nx)
        else:
            errs, endpoint_vec = unroll_trace(prob, x0, y0, args.K_eval, xstar=xstar, nx=nx, net=spec["net"], mode=spec["mode"], L=L, cfg=cfg, fixed=spec["fixed"], counters=counters)
        if DEV == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        per_iter = elapsed / args.K_eval
        for i, e in enumerate(errs, start=1):
            oracle = oracle_marks[i - 1] if oracle_marks is not None else i * (1 + 2 + 1)
            network = i if spec["net"] is not None else 0
            rows.append(
                {
                    "data": data,
                    "blur": blur,
                    "noise": noise,
                    "mode": mode,
                    "c": c,
                    "method": method,
                    "iteration": i,
                    "oracle": oracle,
                "network_forward": network,
                "backtracks": int(counters.get("backtracks", 0) * i / max(1, args.K_eval)) if oracle_marks is not None else 0,
                "wall_time_s": i * per_iter,
                "mean_rel_primal_error": e,
                }
            )
        endpoint = np.asarray(endpoint_vec, dtype=float)
        oracle_K = oracle_marks[-1] if oracle_marks is not None else args.K_eval * (1 + 2 + 1)
        for j, val in enumerate(endpoint_vec):
            per_image_rows.append(
                {
                    "data": data,
                    "blur": blur,
                    "noise": noise,
                    "mode": mode,
                    "c": c,
                    "method": method,
                    "image_index": j,
                    "oracle_K": oracle_K,
                    "network_forward_K": args.K_eval if spec["net"] is not None else 0,
                    "backtracks_K": counters.get("backtracks", 0),
                    "rel_primal_error_K": float(val),
                }
            )
        summary.append(
            {
                "data": data,
                "blur": blur,
                "noise": noise,
                "mode": mode,
                "c": c,
                "method": method,
                "err_K": float(endpoint.mean()),
                "err_K_mean": float(endpoint.mean()),
                "err_K_std": float(endpoint.std(ddof=1)) if endpoint.size > 1 else 0.0,
                "err_K_median": float(np.median(endpoint)),
                "wall_time_K": elapsed,
                **counters,
            }
        )
    return rows, summary, per_image_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image_dir", default="standard_images")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--test_seed", type=int, default=20240704)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--ntest", type=int, default=32)
    ap.add_argument("--K_eval", type=int, default=300)
    ap.add_argument("--ref_iters", type=int, default=2500)
    ap.add_argument("--data", default="real,synthetic")
    ap.add_argument("--blur", default="train,wide,motion")
    ap.add_argument("--noise", default="train,high")
    ap.add_argument("--c_values", default="0.25,0.5,0.9")
    ap.add_argument("--modes", default="bv")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    download_images(args.image_dir)
    all_rows, all_summary, all_per_image = [], [], []
    for data in args.data.split(","):
        for blur in args.blur.split(","):
            for noise in args.noise.split(","):
                for c in [float(x) for x in args.c_values.split(",")]:
                    for mode in args.modes.split(","):
                        print("[eval]", data, blur, noise, "c", c, "mode", mode, flush=True)
                        rows, summary, per_image_rows = evaluate_one(args, data, blur, noise, c, mode)
                        all_rows.extend(rows)
                        all_summary.extend(summary)
                        all_per_image.extend(per_image_rows)
                        with (outdir / "summary_partial.json").open("w", encoding="utf-8") as f:
                            json.dump(all_summary, f, indent=2)
                        with (outdir / "per_image_endpoint_errors_partial.csv").open("w", newline="", encoding="utf-8") as f:
                            fields = [
                                "data",
                                "blur",
                                "noise",
                                "mode",
                                "c",
                                "method",
                                "image_index",
                                "oracle_K",
                                "network_forward_K",
                                "backtracks_K",
                                "rel_primal_error_K",
                            ]
                            w = csv.DictWriter(f, fieldnames=fields)
                            w.writeheader()
                            w.writerows(all_per_image)
    with (outdir / "curves_formal.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "data",
            "blur",
            "noise",
            "mode",
            "c",
            "method",
            "iteration",
            "oracle",
            "network_forward",
            "backtracks",
            "wall_time_s",
            "mean_rel_primal_error",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    with (outdir / "summary_formal.json").open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    with (outdir / "per_image_endpoint_errors.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "data",
            "blur",
            "noise",
            "mode",
            "c",
            "method",
            "image_index",
            "oracle_K",
            "network_forward_K",
            "backtracks_K",
            "rel_primal_error_K",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_per_image)
    print("saved", outdir / "curves_formal.csv")
    print("saved", outdir / "summary_formal.json")
    print("saved", outdir / "per_image_endpoint_errors.csv")


if __name__ == "__main__":
    main()
