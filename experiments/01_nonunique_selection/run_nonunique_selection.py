import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def B(z, omega=1.0):
    out = np.zeros_like(z)
    out[..., 0] = -omega * z[..., 1]
    out[..., 1] = omega * z[..., 0]
    return out


def C(z, c=0.5):
    out = np.zeros_like(z)
    out[..., 2] = c * z[..., 2]
    return out


def prox_A(v):
    # A is the normal cone to {z_3 >= 0}; its resolvent is projection.
    p = v.copy()
    p[..., 2] = np.maximum(p[..., 2], 0.0)
    return p


def fbhf_map(z, gamma):
    p = prox_A(z - gamma * (B(z) + C(z)))
    return p + gamma * (B(z) - B(p))


def run_one(z0, steps, gamma, anchor=None):
    z = z0.astype(float).copy()
    rows = []
    for k in range(steps + 1):
        residual = np.linalg.norm(z - fbhf_map(z, gamma))
        rows.append(
            {
                "iteration": k,
                "distance_to_min_norm": np.linalg.norm(z),
                "residual": residual,
                "free_coordinate": z[3],
                "z1": z[0],
                "z2": z[1],
                "z3": z[2],
                "z4": z[3],
            }
        )
        if k == steps:
            break
        Tz = fbhf_map(z, gamma)
        if anchor is None:
            z = Tz
        else:
            lam = 1.0 / (k + 2.0)
            z = lam * anchor + (1.0 - lam) * Tz
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="nonunique_selection_results")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--gamma", type=float, default=0.35)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    initials = np.array(
        [
            [2.0, -1.0, 2.0, -3.0],
            [-1.5, 2.5, -1.0, -1.0],
            [1.2, 1.8, 1.0, 1.0],
            [-2.0, -1.0, 3.0, 3.0],
        ],
        dtype=float,
    )
    anchor = np.zeros(4)

    curve_rows = []
    final_rows = []
    for init_id, z0 in enumerate(initials):
        for method, anc in [("plain-FBHF", None), ("Halpern-FBHF", anchor)]:
            rows = run_one(z0, args.steps, args.gamma, anchor=anc)
            for r in rows:
                r.update({"method": method, "init_id": init_id, "initial_free_coordinate": z0[3]})
                curve_rows.append(r)
            last = rows[-1]
            final_rows.append(
                {
                    "method": method,
                    "init_id": init_id,
                    "initial_free_coordinate": z0[3],
                    "final_free_coordinate": last["free_coordinate"],
                    "final_distance_to_min_norm": last["distance_to_min_norm"],
                    "final_residual": last["residual"],
                    "z1": last["z1"],
                    "z2": last["z2"],
                    "z3": last["z3"],
                    "z4": last["z4"],
                }
            )

    curves_csv = outdir / "nonunique_curves.csv"
    with curves_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method",
            "init_id",
            "initial_free_coordinate",
            "iteration",
            "distance_to_min_norm",
            "residual",
            "free_coordinate",
            "z1",
            "z2",
            "z3",
            "z4",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(curve_rows)

    finals_csv = outdir / "nonunique_finals.csv"
    with finals_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method",
            "init_id",
            "initial_free_coordinate",
            "final_free_coordinate",
            "final_distance_to_min_norm",
            "final_residual",
            "z1",
            "z2",
            "z3",
            "z4",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(final_rows)

    summary = {
        "problem": "0 in N_{z3>=0}(z) + Bz + Cz, where B rotates (z1,z2), C damps z3, and z4 is a free solution coordinate",
        "solution_set": "{(0,0,0,t): t in R}",
        "minimum_norm_solution": [0.0, 0.0, 0.0, 0.0],
        "gamma": args.gamma,
        "steps": args.steps,
        "plain_final_distance_mean": float(np.mean([r["final_distance_to_min_norm"] for r in final_rows if r["method"] == "plain-FBHF"])),
        "halpern_final_distance_mean": float(np.mean([r["final_distance_to_min_norm"] for r in final_rows if r["method"] == "Halpern-FBHF"])),
    }
    with (outdir / "nonunique_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("saved", curves_csv)
    print("saved", finals_csv)


if __name__ == "__main__":
    main()
