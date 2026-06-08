"""Side-by-side box-plot comparison: RL policy vs. traditional 2-stage method.

For each test (hazards / spread / tasks) both methods are drawn at the same
x-position, offset slightly left (RL) and right (traditional).

RL boxes     — proper quartiles from monte_carlo_analysis.py output CSVs.
Traditional  — approximated from aggregate statistics using a Bernoulli-std
               estimate:  σ = sqrt(p·(1−p)/n) · 100,  where  p = tasks_rescued_pct/100
               and n = num_trials.  q1 ≈ mean − 0.674σ,  q3 ≈ mean + 0.674σ.
               Whiskers extend to 1.5 × IQR, clipped to [0, 100].

Median connection lines
  RL          — red solid  line through RL medians
  Traditional — red dashed line through traditional means

Usage
-----
    python plot_comparison.py                      # auto-discovers files
    python plot_comparison.py \\
        --rl-hazards  mc_results/hazards_results.csv \\
        --tr-hazards  mc_results/hazard_count_success_summary.csv \\
        --rl-spread   mc_results/spread_results.csv \\
        --tr-spread   mc_results/pf_success_summary.csv \\
        --rl-tasks    mc_results/tasks_results.csv \\
        --tr-tasks    mc_results/task_count_success_summary.csv \\
        --out-dir     mc_results
"""

import argparse
import csv
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
RL_COLOR   = "steelblue"
TRAD_COLOR = "coral"
OFFSET     = 0.12    # half-distance between the two box centres per x-tick
BOX_WIDTH  = 0.22
# ─────────────────────────────────────────────


# =========================================================================== #
#  CSV readers                                                                 #
# =========================================================================== #

def _normalise_key(s):
    """Convert a CSV key string to float if numeric, else keep as string.
    This ensures "0.02" and "0.020000" map to the same key (0.02)."""
    try:
        return float(s)
    except ValueError:
        return s


def _read_rl_csv(path):
    """Return dict {normalised_key → bxp_stats_dict (values in %)}.

    RL CSV has rescue_rate_q25 / q75 / median columns.
    """
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = _normalise_key(list(r.values())[0])
            med = float(r["rescue_rate_median"]) * 100
            q1  = float(r["rescue_rate_q25"])   * 100
            q3  = float(r["rescue_rate_q75"])   * 100
            iqr = q3 - q1
            rows[key] = {
                "med":    med,
                "q1":     q1,
                "q3":     q3,
                "whislo": max(0.0,   q1 - 1.5 * iqr),
                "whishi": min(100.0, q3 + 1.5 * iqr),
                "fliers": [],
            }
    return rows


def _read_trad_csv(path):
    """Return dict {normalised_key → bxp_stats_dict (values in %)}.

    Expects a raw per-trial CSV where the last column is the parameter being
    varied (e.g. hazard_count, pf_value, task_count) and tasks_rescued_pct
    holds the per-trial rescue percentage.  Proper quartiles are computed
    directly from the raw data — no approximation needed.
    """
    # Collect per-trial values grouped by the parameter column (last column)
    groups: dict = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        param_col = reader.fieldnames[-1]          # last column = varied parameter
        for r in reader:
            key = _normalise_key(r[param_col])
            groups.setdefault(key, []).append(float(r["tasks_rescued_pct"]))

    rows = {}
    for key, vals in groups.items():
        arr    = np.array(vals, dtype=np.float32)
        med    = float(np.median(arr))
        q1     = float(np.percentile(arr, 25))
        q3     = float(np.percentile(arr, 75))
        iqr    = q3 - q1
        rows[key] = {
            "med":    med,
            "q1":     q1,
            "q3":     q3,
            "whislo": max(0.0,   q1 - 1.5 * iqr),
            "whishi": min(100.0, q3 + 1.5 * iqr),
            "fliers": [],
        }
    return rows


# =========================================================================== #
#  Comparison plot                                                             #
# =========================================================================== #

def _comparison_plot(rl_rows, trad_rows, xlabel, title, out_path, label_fmt="{}"):
    # Union of keys, sorted numerically where possible
    all_keys = sorted(
        set(rl_rows) | set(trad_rows),
        key=lambda x: float(x) if _is_numeric(x) else x,
    )
    n     = len(all_keys)
    x_int = np.arange(1, n + 1)

    rl_pos   = x_int - OFFSET
    trad_pos = x_int + OFFSET

    fig, ax = plt.subplots(figsize=(max(5, n * 2.2), 5))

    def _draw_boxes(rows, positions, color, label):
        stats_list = []
        valid_pos  = []
        for k, xp in zip(all_keys, positions):
            if k in rows:
                stats_list.append(rows[k])
                valid_pos.append(xp)
        if not stats_list:
            return []
        return ax.bxp(
            stats_list,
            positions   = valid_pos,
            widths      = BOX_WIDTH,
            patch_artist= True,
            showfliers  = False,
            boxprops    = dict(facecolor=color, alpha=0.65, edgecolor="black", linewidth=1.1),
            medianprops = dict(color="orange",  linewidth=2.0),
            whiskerprops= dict(color="black",   linewidth=1.0),
            capprops    = dict(color="black",   linewidth=1.0),
        )

    _draw_boxes(rl_rows,   rl_pos,   RL_COLOR,   "RL policy")
    _draw_boxes(trad_rows, trad_pos, TRAD_COLOR, "Traditional")

    # ── Median / mean connection lines ─────────────────────────────────────
    def _median_line(rows, positions, linestyle):
        pts = [(xp, rows[k]["med"])
               for k, xp in zip(all_keys, positions) if k in rows]
        if len(pts) < 2:
            return
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="red", linestyle=linestyle, linewidth=1.6,
                marker="o", markersize=4, zorder=6)

    _median_line(rl_rows,   rl_pos,   "-")
    _median_line(trad_rows, trad_pos, "--")

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_xticks(x_int)
    ax.set_xticklabels([_fmt_label(k, label_fmt) for k in all_keys])
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Task rescue rate (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.grid(True, axis="y", alpha=0.35)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=RL_COLOR,   alpha=0.65, edgecolor="black",
                       label="RL policy"),
        mpatches.Patch(facecolor=TRAD_COLOR, alpha=0.65, edgecolor="black",
                       label="Traditional method"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _fmt_label(k, fmt):
    """Format a CSV key string using fmt; converts to float first if needed."""
    try:
        return fmt.format(float(k))
    except (ValueError, TypeError):
        return fmt.format(k)


# =========================================================================== #
#  Auto-discovery helpers                                                      #
# =========================================================================== #

def _auto(default, *candidates):
    """Return the first candidate path that exists, else the default."""
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return default


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare RL policy vs. traditional method via box plots")
    p.add_argument("--rl-hazards",  default=None)
    p.add_argument("--tr-hazards",  default=None)
    p.add_argument("--rl-spread",   default=None)
    p.add_argument("--tr-spread",   default=None)
    p.add_argument("--rl-tasks",    default=None)
    p.add_argument("--tr-tasks",    default=None)
    p.add_argument("--out-dir",     default="mc_results")
    return p.parse_args()


def main():
    args = parse_args()
    D    = args.out_dir

    # Auto-discover files if not provided
    rl_hazards = _auto(args.rl_hazards, args.rl_hazards,
                       os.path.join(D, "hazards_results.csv"))
    tr_hazards = _auto(args.tr_hazards, args.tr_hazards,
                       os.path.join(D, "hazard_count_vs_success_raw.csv"))

    rl_spread  = _auto(args.rl_spread, args.rl_spread,
                       os.path.join(D, "spread_results.csv"))
    tr_spread  = _auto(args.tr_spread, args.tr_spread,
                       os.path.join(D, "pf_vs_success_raw.csv"))

    rl_tasks   = _auto(args.rl_tasks, args.rl_tasks,
                       os.path.join(D, "tasks_results.csv"))
    tr_tasks   = _auto(args.tr_tasks, args.tr_tasks,
                       os.path.join(D, "task_count_vs_success_raw.csv"))

    # ── Hazard count comparison ─────────────────────────────────────────────
    if rl_hazards and tr_hazards and os.path.isfile(rl_hazards) and os.path.isfile(tr_hazards):
        print(f"Hazards:  RL={rl_hazards}  Traditional={tr_hazards}")
        _comparison_plot(
            _read_rl_csv(rl_hazards),
            _read_trad_csv(tr_hazards),
            xlabel    = "Number of initial hazards",
            title     = "Task rescue rate vs. number of hazards\n(2 robots, 2 tasks)",
            out_path  = os.path.join(D, "comparison_hazards.png"),
        )
    else:
        print(f"Skipping hazards comparison (missing files: {rl_hazards}, {tr_hazards})")

    # ── Fire spread rate comparison ─────────────────────────────────────────
    if rl_spread and tr_spread and os.path.isfile(rl_spread) and os.path.isfile(tr_spread):
        print(f"Spread:   RL={rl_spread}  Traditional={tr_spread}")
        _comparison_plot(
            _read_rl_csv(rl_spread),
            _read_trad_csv(tr_spread),
            xlabel    = "Fire spread probability",
            title     = "Task rescue rate vs. fire spread rate\n(2 robots, 2 tasks)",
            out_path  = os.path.join(D, "comparison_spread.png"),
            label_fmt = "{:.3f}",
        )
    else:
        print(f"Skipping spread comparison (missing files: {rl_spread}, {tr_spread})")

    # ── Task count comparison ───────────────────────────────────────────────
    if rl_tasks and tr_tasks and os.path.isfile(rl_tasks) and os.path.isfile(tr_tasks):
        print(f"Tasks:    RL={rl_tasks}  Traditional={tr_tasks}")
        _comparison_plot(
            _read_rl_csv(rl_tasks),
            _read_trad_csv(tr_tasks),
            xlabel    = "Number of tasks",
            title     = "Task rescue rate vs. number of tasks\n(2 robots, matched policy per task count)",
            out_path  = os.path.join(D, "comparison_tasks.png"),
        )
    else:
        print(f"Skipping tasks comparison (missing files: {rl_tasks}, {tr_tasks})")

    print("Done.")


if __name__ == "__main__":
    main()
