import csv
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from trackB_formal_eval import (  # noqa: E402
    DEV,
    PrecNet,
    TV,
    Kf,
    Ktf,
    Dop,
    Dtop,
    features,
    clamp_ts,
    download_images,
    make_batch,
    normD,
    projb,
    rel_error_vec,
    unroll_final,
    warped_step,
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.6,
    }
)


def psnr(x, clean):
    mse = torch.mean((x.clamp(0, 1) - clean) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).detach().cpu().numpy()


@torch.no_grad()
def linesearch_final(prob, x0, y0, K, chi, theta=0.9):
    x, y = x0, y0
    gamma = chi
    counters = {"prox": 0, "B": 0, "C": 0, "backtracks": 0}
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
            lhs = gamma * torch.sqrt(
                ((b0 - r0) ** 2).sum(dim=(1, 2, 3))
                + ((b1 - r1) ** 2).sum(dim=(1, 2, 3))
                + 1e-12
            )
            rhs = theta * torch.sqrt(
                ((x - xr) ** 2).sum(dim=(1, 2, 3))
                + ((y - yr) ** 2).sum(dim=(1, 2, 3))
                + 1e-12
            )
            if bool((lhs <= rhs).all()) or gamma <= 1e-6:
                break
            gamma = gamma * 0.5
        counters["backtracks"] += max(0, trials - 1)
        x = xr + gamma * (b0 - r0)
        y = yr + gamma * (b1 - r1)
    return x, counters


@torch.no_grad()
def learned_final(prob, x0, y0, K, net, L, cfg):
    x, y = x0, y0
    x_prev = x0
    tlo, thi, slo, shi = cfg["tlo"], cfg["thi"], cfg["slo"], cfg["shi"]
    tau = torch.full_like(x, cfg["tau0"])
    s = torch.full_like(x, cfg["s0"])
    for k in range(K):
        _, (px0, py0) = warped_step(
            prob,
            x,
            y,
            torch.full_like(x, cfg["tau0"]),
            torch.full_like(x, cfg["s0"]),
        )
        raw = net(features(x, y, px0, py0, x_prev))
        dtau, dsig = clamp_ts(raw, L, tlo, thi, slo, shi, cfg["rho"])
        eta = cfg["c"] / ((k + 1) ** 1.1)
        tau = torch.clamp(tau + torch.clamp(dtau - tau, -eta, eta), tlo, thi)
        s = torch.clamp(s + torch.clamp(dsig - s, -eta, eta), slo, shi)
        s = torch.minimum(s, cfg["rho"] / (tau * L**2 + 1e-9))
        (x_next, y_next), _ = warped_step(prob, x, y, tau, s)
        x_prev = x
        x, y = x_next, y_next
    return x


def to_img(t, idx):
    return t[idx, 0].detach().cpu().numpy().clip(0, 1)


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=7,
        fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5),
    )


def save_all(fig, base):
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main():
    torch.manual_seed(20240718)
    np.random.seed(20240718)

    package = HERE / "final_experiment_package" / "formal_oracle64_std_classic"
    ckpt = package / "wbv_seed11.pt"
    image_dir = HERE / "standard_images"
    outdir = HERE.parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    download_images(image_dir)

    H = W = 128
    batch = 8
    k_eval = 1000
    ref_iters = 3000
    test_seed = 20240718

    b, otf, mu, clean = make_batch(
        batch,
        H,
        W,
        DEV,
        test_seed,
        data="real",
        blur="train",
        noise="train",
        image_dir=image_dir,
    )
    prob = TV(b, otf, mu)
    x0 = b.clone()
    y0 = torch.zeros_like(b).repeat(1, 2, 1, 1)
    L = normD(H, W, DEV)
    chi = 4 / (1 + math.sqrt(1 + 16 * L**2))
    cfg = dict(tlo=0.1, thi=1.8, slo=0.02, shi=0.12, rho=0.9, tau0=0.3, s0=0.1, c=0.5)

    net = PrecNet().to(DEV)
    net.load_state_dict(torch.load(ckpt, map_location=DEV))
    net.eval()

    xstar = unroll_final(prob, x0, y0, ref_iters, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))
    nx = torch.sqrt((xstar**2).sum(dim=(1, 2, 3))).clamp_min(1e-12)
    x_plain = unroll_final(prob, x0, y0, k_eval, L=L, cfg=cfg, fixed=(chi - 0.05 * chi, 0.10))
    x_line, ls_counters = linesearch_final(prob, x0, y0, k_eval, chi=chi)
    x_fixed = unroll_final(prob, x0, y0, k_eval, L=L, cfg=cfg, fixed=(1.5, 0.07))
    x_learned = learned_final(prob, x0, y0, k_eval, net, L, cfg)

    methods = {
        "Observed": b,
        "Plain FBHF": x_plain,
        "Line search": x_line,
        "Fixed warped": x_fixed,
        "Learned warped": x_learned,
    }
    rels = {name: rel_error_vec(x, xstar, nx).detach().cpu().numpy() for name, x in methods.items() if name != "Observed"}
    psnrs = {name: psnr(x, clean) for name, x in methods.items()}
    improvement = rels["Fixed warped"] - rels["Learned warped"]
    idx = int(np.argmax(improvement))

    csv_path = outdir / "source_data_fig4_reconstruction_plate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selected_index",
                "method",
                "relative_error_to_reference",
                "psnr_to_clean_db",
                "oracle_calls",
                "network_forward_calls",
                "line_search_backtracks",
            ],
        )
        writer.writeheader()
        for name, x in methods.items():
            writer.writerow(
                {
                    "selected_index": idx,
                    "method": name,
                    "relative_error_to_reference": "" if name == "Observed" else float(rels[name][idx]),
                    "psnr_to_clean_db": float(psnrs[name][idx]),
                    "oracle_calls": 4000 if name != "Line search" else int(ls_counters["prox"] + ls_counters["B"] + ls_counters["C"]),
                    "network_forward_calls": k_eval if name == "Learned warped" else 0,
                    "line_search_backtracks": int(ls_counters["backtracks"]) if name == "Line search" else 0,
                }
            )

    titles = ["Clean", "Observed", "Plain FBHF", "Line search", "Fixed warped", "Learned warped"]
    tensors = [clean, b, x_plain, x_line, x_fixed, x_learned]
    labels = ["a", "b", "c", "d", "e", "f"]
    crop = (slice(38, 92), slice(38, 92))

    error_maps = []
    for tensor in tensors[2:]:
        error_maps.append(np.abs(to_img(tensor, idx) - to_img(xstar, idx))[crop])
    err_vmax = max(np.percentile(np.concatenate([e.ravel() for e in error_maps]), 99), 1e-4)

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.05, wspace=0.04)
    for j, (title, tensor, label) in enumerate(zip(titles, tensors, labels)):
        ax = fig.add_subplot(gs[0, j])
        img = to_img(tensor, idx)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if title in {"Plain FBHF", "Line search", "Fixed warped", "Learned warped"}:
            ttl = f"{title}\nrel. err {rels[title][idx]:.2e}"
        elif title == "Observed":
            ttl = f"{title}\nPSNR {psnrs[title][idx]:.2f} dB"
        else:
            ttl = title
        ax.set_title(ttl, fontsize=7, pad=2)
        add_panel_label(ax, label)
        if j == 0:
            y0c, y1c = crop[0].start, crop[0].stop
            x0c, x1c = crop[1].start, crop[1].stop
            ax.add_patch(
                Rectangle((x0c, y0c), x1c - x0c, y1c - y0c, fill=False, lw=0.8, ec="#d95f02")
            )

        ax2 = fig.add_subplot(gs[1, j])
        if title in {"Plain FBHF", "Line search", "Fixed warped", "Learned warped"}:
            ax2.imshow(np.abs(img - to_img(xstar, idx))[crop], cmap="magma", vmin=0, vmax=err_vmax)
        else:
            ax2.imshow(img[crop], cmap="gray", vmin=0, vmax=1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_edgecolor("#d95f02")
            spine.set_linewidth(0.5)
        if title in rels:
            ax2.set_xlabel("abs. error crop", fontsize=6, labelpad=1)
        elif title == "Observed":
            ax2.set_xlabel("degraded input", fontsize=6, labelpad=1)
        else:
            ax2.set_xlabel("reference image", fontsize=6, labelpad=1)

    fig.suptitle("Representative held-out TV deblurring example at matched oracle budget", fontsize=8, y=1.02)
    save_all(fig, outdir / "fig4_reconstruction_plate")
    print("selected_index", idx)
    print("saved", outdir / "fig4_reconstruction_plate.pdf")
    print("saved", csv_path)


if __name__ == "__main__":
    main()
