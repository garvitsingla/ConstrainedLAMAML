import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Font sizes tuned for 4-per-row in a single-column LaTeX paper ───────────
mpl.rcParams.update({
    "font.size":         22,   # base fallback
    "axes.titlesize":    22,   # plot title (env name)
    "axes.labelsize":    20,   # x/y axis labels
    "xtick.labelsize":   16,   # tick numbers
    "ytick.labelsize":   16,
    "axes.linewidth":    2,
    "lines.linewidth":   2,
    "font.family":       "sans-serif",
    "figure.dpi":        300,
    "xtick.major.pad":   2,
    "ytick.major.pad":   2,
    "axes.labelpad":     2,
})

# ── Color palette — consistent across all envs ──────────────────────────────
COLORS = {
    "c_lamaml":         "#2ca02c",   # green
    "crl":              "#d62728",   # red
}


def safe_load_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def maybe_smooth(y: np.ndarray, w: int):
    if w <= 1:
        return y
    kernel = np.ones(w) / w
    y_pad  = np.pad(y, (w - 1, 0), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env",              type=str,   required=True)
    p.add_argument("--metrics_root",     type=str,   default="metrics")
    p.add_argument("--out_dir",          type=str,   default="plots")
    p.add_argument("--format",           type=str,   default="png",
                   choices=["png", "pdf", "svg"])
    p.add_argument("--smooth",           type=int,   default=0)
    p.add_argument("--delta-g",          type=float, default=0.3)
    p.add_argument("--delta-c",          type=float, default=0.1)
    args = p.parse_args()

    env      = args.env
    base_dir = os.path.join(args.metrics_root, env)
    w        = args.smooth if args.smooth > 0 else 10

    # ── File paths ───────────────────────────────────────────────────────────
    paths = {
        "c_lamaml": (
            os.path.join(base_dir, f"c_lamaml_avg_steps_dt{args.delta_g}_dc{args.delta_c}.npy"),
            os.path.join(base_dir, f"c_lamaml_meta_dt{args.delta_g}_dc{args.delta_c}.json"),
        ),
        "crl": (
            os.path.join(base_dir, "crl_avg_steps.npy"),
            os.path.join(base_dir, "crl_meta.json"),
        ),
    }

    paths_std = {
        "c_lamaml":         os.path.join(base_dir, f"c_lamaml_std_steps_dt{args.delta_g}_dc{args.delta_c}.npy"),
        "crl":              os.path.join(base_dir, "crl_std_steps.npy"),
    }

    paths_costs = {
        "c_lamaml":         os.path.join(base_dir, f"c_lamaml_avg_costs_dt{args.delta_g}_dc{args.delta_c}.npy"),
        "crl":              os.path.join(base_dir, "crl_avg_costs.npy"),
    }

    paths_std_costs = {
        "c_lamaml":         os.path.join(base_dir, f"c_lamaml_std_costs_dt{args.delta_g}_dc{args.delta_c}.npy"),
        "crl":              os.path.join(base_dir, "crl_std_costs.npy"),
    }

    # ── Load ─────────────────────────────────────────────────────────────────
    series     = {}
    series_std = {}
    series_costs = {}
    series_std_costs = {}
    meta       = {}

    for key, (npy_path, json_path) in paths.items():
        if os.path.exists(npy_path):
            series[key] = np.load(npy_path)
            meta[key]   = safe_load_json(json_path)
        else:
            print(f"Skipping {key}, missing: {npy_path}")

    for key, npy_path in paths_std.items():
        if os.path.exists(npy_path):
            series_std[key] = np.load(npy_path)

    for key, npy_path in paths_costs.items():
        if os.path.exists(npy_path):
            series_costs[key] = np.load(npy_path)

    for key, npy_path in paths_std_costs.items():
        if os.path.exists(npy_path):
            series_std_costs[key] = np.load(npy_path)

    # ── Labels ───────────────────────────────────────────────────────────────
    label_defaults = {
        "c_lamaml":         "C-LAMAML",
        "crl":              "CRL",
    }
    labels = {k: meta[k].get("label", label_defaults[k])
              for k in label_defaults if k in meta}

    # ── PLOTS: Mean +/- std — NO legend ─────────────────────────────────────
    if series_std:
        out_var_dir = os.path.join(args.out_dir, env)
        os.makedirs(out_var_dir, exist_ok=True)

        fig, ax      = plt.subplots(figsize=(5, 3.5))
        line_handles = []
        line_labels  = []

        for key in labels:
            if key in series and key in series_std:
                y_mean = maybe_smooth(series[key].astype(float), w)
                y_std  = maybe_smooth(series_std[key].astype(float), w)
                x      = np.arange(1, len(y_mean) + 1)
                color  = COLORS.get(key)

                line, = ax.plot(x, y_mean, label=labels[key], color=color)
                ax.fill_between(x,
                                y_mean - y_std,
                                y_mean + y_std,
                                color=color, alpha=0.2)
                line_handles.append(line)
                line_labels.append(labels[key])

        ax.set_xlabel("Meta-iteration")
        ax.set_ylabel("Average Steps")
        ax.set_title(env)
        ax.grid(True, alpha=0.4)
        fig.tight_layout(pad=0.1)

        out_var_path = os.path.join(out_var_dir, f"dt{args.delta_g}_dc{args.delta_c}.{args.format}")
        fig.savefig(out_var_path, dpi=300, bbox_inches="tight")
        print(f"Saved variance plot: {out_var_path}")
        plt.close(fig)

        # ── Shared horizontal legend — saved ONCE ────────────────────────────
        legend_path = os.path.join(out_var_dir, f"legend.{args.format}")
        if not os.path.exists(legend_path) and line_handles:
            fig_leg, ax_leg = plt.subplots(figsize=(9, 0.5))
            ax_leg.axis("off")
            ax_leg.legend(
                handles=line_handles,
                labels=line_labels,
                loc="center",
                ncol=len(line_handles),   # all 4 on one row
                frameon=True,
                fontsize=13,
                handlelength=2.2,
                handletextpad=0.6,
                columnspacing=2.0,
            )
            fig_leg.tight_layout(pad=0.1)
            fig_leg.savefig(legend_path, dpi=300, bbox_inches="tight")
            print(f"Saved legend: {legend_path}")
            plt.close(fig_leg)

    if series_std_costs:
        out_var_dir = os.path.join(args.out_dir, env)
        os.makedirs(out_var_dir, exist_ok=True)

        fig, ax      = plt.subplots(figsize=(5, 3.5))
        line_handles = []
        line_labels  = []

        for key in labels:
            if key in series_costs and key in series_std_costs:
                y_mean = maybe_smooth(series_costs[key].astype(float), w)
                y_std  = maybe_smooth(series_std_costs[key].astype(float), w)
                x      = np.arange(1, len(y_mean) + 1)
                color  = COLORS.get(key)

                line, = ax.plot(x, y_mean, label=labels[key], color=color)
                ax.fill_between(x,
                                y_mean - y_std,
                                y_mean + y_std,
                                color=color, alpha=0.2)
                line_handles.append(line)
                line_labels.append(labels[key])

        ax.set_xlabel("Meta-iteration")
        ax.set_ylabel("Average Cost")
        ax.set_title(env)
        ax.grid(True, alpha=0.4)
        fig.tight_layout(pad=0.1)

        out_var_path = os.path.join(out_var_dir, f"dt{args.delta_g}_dc{args.delta_c}_costs.{args.format}")
        fig.savefig(out_var_path, dpi=300, bbox_inches="tight")
        print(f"Saved cost plot: {out_var_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()