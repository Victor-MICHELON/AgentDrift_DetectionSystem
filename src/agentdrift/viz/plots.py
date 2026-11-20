"""Figures of the EDA and the report: one style, one colour per entity, export to PNG and PDF.

Colour rules (validated categorical palette, fixed order, colour-blind safe on adjacent pairs):
- each entity keeps its colour in every figure (labels, source categories, domains, datasets);
- scatter plots with more than three groups are faceted (one highlighted group per panel);
- magnitudes use a single-hue sequential ramp; text is always dark or white ink, never a series colour;
- three slots are below 3:1 contrast on white, so every figure has a legend or direct labels and the
  notebook shows the underlying table next to it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyBboxPatch, Patch

from agentdrift.config import FIGURES, ROOT

REPORT_FIGURES = ROOT / "report" / "latex" / "figures"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE, MUTED = "#e4e3df", "#ffffff", "#d6d5d0"
SEQUENTIAL = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("agentdrift_blue", SEQUENTIAL)

LABEL_ORDER = ["benign", "injection_point", "hijacked", "failed_injection"]
SOURCE_ORDER = ["benign", "hard_negative", "failed_attack", "attacked_full", "attacked_partial", "attacked_delayed"]
DOMAIN_ORDER = ["email", "banking", "web", "coding", "medical"]
LABEL_COLORS = dict(zip(LABEL_ORDER, SERIES))
SOURCE_COLORS = dict(zip(SOURCE_ORDER, SERIES))
DOMAIN_COLORS = dict(zip(DOMAIN_ORDER, SERIES))
DATASET_COLORS = {"AgentDrift": SERIES[0], "ATBench": SERIES[1]}
PRETTY = {
    "benign": "benign", "injection_point": "injection point", "hijacked": "hijacked",
    "failed_injection": "failed injection", "hard_negative": "hard negative", "failed_attack": "failed attack",
    "attacked_full": "attack · full", "attacked_partial": "attack · partial", "attacked_delayed": "attack · delayed",
}


def pretty(x: str) -> str:
    return PRETTY.get(x, x)


def setup() -> None:
    """Global matplotlib style: recessive axes and grid, dark ink, white surface."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
            "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.labelsize": 9.5, "axes.labelcolor": INK_2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
            "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
            "text.color": INK, "legend.frameon": False, "legend.fontsize": 8.5,
            "lines.linewidth": 2, "patch.linewidth": 0,
        }
    )


def save(fig: plt.Figure, name: str, report: bool = True) -> Path:
    """Write results/figures/<name>.png and, for report figures, report/latex/figures/<name>.pdf."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{name}.png"
    fig.savefig(png)
    if report:
        REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(REPORT_FIGURES / f"{name}.pdf")
    return png


def _lighten(color: str, amount: float) -> tuple:
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _ink_on(rgba) -> str:
    r, g, b = rgba[:3]
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


def _legend_top(ax, handles, ncol=None, y=1.02):
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, y), ncol=ncol or len(handles),
              handlelength=1.0, handleheight=0.8, columnspacing=1.2, borderaxespad=0)


def _patches(colors: dict, keys) -> list:
    return [Patch(facecolor=colors[k], label=pretty(k)) for k in keys]


def heatmap(ax, table: pd.DataFrame, fmt: str = "{:.0%}", vmin=0.0, vmax=None, cbar: bool = False,
            xlabel: str = "", ylabel: str = "", na_text: str = "") -> None:
    """Annotated heatmap on a single-hue ramp; NaN cells left blank."""
    data = table.to_numpy(dtype=float)
    vmax = np.nanmax(data) if vmax is None else vmax
    im = ax.imshow(np.ma.masked_invalid(data), cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [textwrap.fill(pretty(str(c)), 16) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [pretty(str(i)) for i in table.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, table.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = na_text if np.isnan(v) else ("<1%" if fmt.endswith("%}") and 0 < v < 0.005 else fmt.format(v))
            color = INK_3 if np.isnan(v) else _ink_on(im.cmap(im.norm(v)))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if cbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)


def _hbar_stack(ax, table: pd.DataFrame, colors: dict, keys: list, normalize: bool = False):
    t = table[keys].astype(float)
    if normalize:
        t = t.div(t.sum(axis=1), axis=0)
    left = np.zeros(len(t))
    y = np.arange(len(t))
    for k in keys:
        ax.barh(y, t[k], left=left, color=colors[k], height=0.62, edgecolor=SURFACE, linewidth=1.5)
        left += t[k].to_numpy()
    ax.set_yticks(y, [pretty(str(i)) for i in t.index])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    return left


# --------------------------------------------------------------------------- A1


def plot_category_by_domain(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    totals = _hbar_stack(ax, table, SOURCE_COLORS, SOURCE_ORDER)
    for yi, tot in enumerate(totals):
        ax.text(tot + 30, yi, f"{int(tot):,}", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("trajectories")
    ax.set_xlim(0, totals.max() * 1.1)
    _legend_top(ax, _patches(SOURCE_COLORS, SOURCE_ORDER), ncol=6)
    ax.set_title("Trajectories per domain and source category", pad=26)
    return fig


def plot_example_trajectories(ex: pd.DataFrame) -> plt.Figure:
    """One row per source category, one box per step: tool name, coloured by step label."""
    cats = list(ex["source_category"].cat.categories.intersection(ex["source_category"].unique()))
    cats = [c for c in SOURCE_ORDER if c in cats]
    n_t = int(ex["t"].max()) + 1
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(cats) + 1.0))
    for r, cat in enumerate(cats):
        rows = ex[ex["source_category"] == cat]
        for _, s in rows.iterrows():
            color = LABEL_COLORS[s["label"]]
            ax.add_patch(FancyBboxPatch((s["t"] + 0.04, r - 0.36), 0.92, 0.72, boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=_lighten(color, 0.72), edgecolor=color, linewidth=1.6))
            ax.text(s["t"] + 0.5, r, s["tool"].replace("_", " "), ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, n_t)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.set_yticks(range(len(cats)), [pretty(c) for c in cats])
    ax.set_xticks(np.arange(n_t) + 0.5, [f"t = {t}" for t in range(n_t)])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    _legend_top(ax, _patches(LABEL_COLORS, LABEL_ORDER), y=1.0)
    ax.set_title("Example trajectories (train): proposed tool per step, coloured by step label", pad=24)
    return fig


# --------------------------------------------------------------------------- A3


def plot_prevalence(by_split: pd.DataFrame, by_domain: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, gridspec_kw={"width_ratios": [4, 5]})
    targets = {"y_scan": SERIES[0], "y_gate": SERIES[1]}
    for ax, t, title in ((axes[0], by_split, "by split"), (axes[1], by_domain, "by domain")):
        t = t.drop(index="all", errors="ignore")
        x = np.arange(len(t))
        for k, (col, color) in enumerate(targets.items()):
            bars = ax.bar(x + (k - 0.5) * 0.36, t[col], width=0.34, color=color, label=col)
            for b, v in zip(bars, t[col]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(x, [str(i) for i in t.index])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"Positive rate per step, {title}", fontsize=10)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        ax.set_ylim(0, 0.22)
    fig.legend(handles=_patches({"y_scan": SERIES[0], "y_gate": SERIES[1]}, ["y_scan", "y_gate"]),
               loc="upper right", ncol=2, bbox_to_anchor=(1, 1.04))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- A4


def plot_length_by_category(table: pd.DataFrame) -> plt.Figure:
    counts = table.drop(columns="mean")
    share = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    heatmap(ax, share, xlabel="number of steps in the trajectory")
    for i, m in enumerate(table["mean"]):
        ax.text(share.shape[1] - 0.4, i, f"mean {m:.2f}", va="center", ha="left", fontsize=8, color=INK_2)
    ax.set_title("Trajectory length by source category (row share)")
    return fig


def plot_hijack_rate_by_position(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    heatmap(ax, table, xlabel="step index t (0-based)", na_text="–")
    ax.set_title("Share of hijacked steps by position and source category: P(y_gate = 1 | t)")
    return fig


def plot_injection_and_gap(position: pd.DataFrame, gap: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0))
    heatmap(axes[0], position, fmt="{:,.0f}", xlabel="injection step t", vmax=position.to_numpy().max())
    axes[0].set_title("Position of the poisoned observation (count)")
    heatmap(axes[1], gap, fmt="{:,.0f}", xlabel="steps from injection to first hijacked step")
    axes[1].set_title("Injection → first hijacked step (count)")
    fig.tight_layout(w_pad=3)
    return fig


# --------------------------------------------------------------------------- A5


def plot_tools(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 0.24 * len(table) + 1.2))
    _hbar_stack(ax, table, LABEL_COLORS, LABEL_ORDER, normalize=True)
    for yi, tot in enumerate(table["total"]):
        ax.text(1.01, yi, f"{int(tot):,}", va="center", fontsize=7.5, color=INK_2, transform=ax.get_yaxis_transform())
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the tool's steps (right: number of steps)")
    _legend_top(ax, _patches(LABEL_COLORS, LABEL_ORDER))
    ax.set_title(f"Label mix of the {len(table)} most used tools", pad=26)
    return fig


def plot_share_bars(series: pd.Series, title: str, xlabel: str, color: str = SERIES[0]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(series) + 1.1))
    y = np.arange(len(series))
    ax.barh(y, series.to_numpy(), color=color, height=0.6)
    for yi, v in zip(y, series):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [pretty(str(i)) for i in series.index])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(1.0, series.max() * 1.12))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return fig


# --------------------------------------------------------------------------- A6 / A7 (heatmaps)


def plot_table_heatmap(table: pd.DataFrame, title: str, xlabel: str = "", ylabel: str = "",
                       fmt: str = "{:.0%}", width: float = 8.5) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(width, 0.42 * len(table) + 1.3))
    heatmap(ax, table, fmt=fmt, xlabel=xlabel, ylabel=ylabel, vmax=1.0 if fmt.endswith("%}") else None)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=0)
    for lab in ax.get_xticklabels():
        lab.set_wrap(True)
    return fig


# --------------------------------------------------------------------------- A8


def plot_world_purity(purity: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(11, 2.4), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, dom in zip(axes, DOMAIN_ORDER):
        x = purity.loc[purity["domain"] == dom, "majority_share"]
        ax.hist(x, bins=bins, weights=purity.loc[x.index, "n"], color=DOMAIN_COLORS[dom], edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(dom, fontsize=10)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("trajectories")
    fig.supxlabel("share of the world's trajectories in its majority source category", fontsize=9, color=INK_2)
    fig.suptitle("World identity predicts the category (each bar weighted by trajectories)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_shortcuts(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    s = table["roc_auc"].astype(float).sort_values()
    y = np.arange(len(s))
    ax.barh(y, s - 0.5, left=0.5, color=SERIES[1], height=0.55)
    for yi, v in zip(y, s):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0.5, color=INK_3, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, -0.55, "chance", color=INK_3, fontsize=8, ha="left", va="center")
    ax.set_yticks(y, s.index)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.8, len(s) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC on val (fitted on train)")
    ax.set_title("Shortcut baselines: no agent content, only metadata")
    return fig


def plot_associations(table: pd.DataFrame) -> plt.Figure:
    t = table.assign(name=table["variable"] + " → " + table["target"]).set_index("name")["cramers_v"].sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(t) + 1.1))
    y = np.arange(len(t))
    ax.barh(y, t, color=SERIES[0], height=0.55)
    for yi, v in zip(y, t):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cramér's V (descriptive, steps are not independent)")
    ax.set_title("Association between candidate variables and labels")
    return fig


# --------------------------------------------------------------------------- A9


def plot_projection(pts: pd.DataFrame) -> plt.Figure:
    """Faceted t-SNE: one highlighted group per panel (labels on row 1, domains on row 2)."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    panels = [("label", k, LABEL_COLORS[k]) for k in LABEL_ORDER] + [None] + [
        ("domain", k, DOMAIN_COLORS[k]) for k in DOMAIN_ORDER
    ]
    for ax, panel in zip(axes.ravel(), panels):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if panel is None:
            continue
        col, key, color = panel
        sel = pts[col] == key
        ax.scatter(pts.loc[~sel, "x"], pts.loc[~sel, "y"], s=3, color=MUTED, linewidths=0)
        ax.scatter(pts.loc[sel, "x"], pts.loc[sel, "y"], s=5, color=color, linewidths=0)
        ax.set_title(f"{col}: {pretty(key)}", fontsize=9.5, fontweight="normal")
    fig.suptitle("t-SNE of step text (tool + arguments + observation), descriptive only", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- A10


def plot_external(traj: pd.DataFrame, atb: pd.DataFrame) -> plt.Figure:
    pairs = [
        ("agent actions per trajectory", traj["n_steps"], atb["n_actions"], np.arange(0.5, 32.5, 1)),
        ("task length (characters)", traj["task"].str.len(), atb["task_chars"], np.linspace(0, 900, 37)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.9))
    for ax, (title, a, b, bins) in zip(axes, pairs):
        for name, x in (("AgentDrift", a), ("ATBench", b)):
            ax.hist(x.clip(upper=bins[-1]), bins=bins, density=True, histtype="step", linewidth=2,
                    color=DATASET_COLORS[name], label=f"{name} (n = {len(x):,})")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("density")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    fig.suptitle("External test set: ATBench vs AgentDrift", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig
