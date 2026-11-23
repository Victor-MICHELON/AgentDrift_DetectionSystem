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
        ("tool calls per trajectory", traj["n_steps"], atb["n_actions"], np.arange(-0.5, 23.5, 1)),
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


# --------------------------------------------------------------------------- part B: effect of the preprocessing

VERSION_COLORS = {"raw": INK_3, "anonymised": SERIES[0], "masked": SERIES[2]}


def plot_before_after_bars(table: pd.DataFrame, title: str, xlabel: str, reference: tuple[str, float] | None = None
                           ) -> plt.Figure:
    """Paired horizontal bars, one colour per text version (columns of `table` among raw / anonymised / masked);
    optional dashed reference line (label, value), e.g. a chance level."""
    versions = [v for v in VERSION_COLORS if v in table.columns]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(table) + 1.3))
    h = 0.8 / len(versions)
    y = np.arange(len(table))
    for k, v in enumerate(versions):
        vals = table[v].astype(float).to_numpy()
        ax.barh(y + (k - (len(versions) - 1) / 2) * h, vals, height=h * 0.9, color=VERSION_COLORS[v])
        for yi, val in zip(y + (k - (len(versions) - 1) / 2) * h, vals):
            ax.text(val + 0.008, yi, "0" if val == 0 else ("<1%" if val < 0.005 else f"{val:.0%}"),
                    va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(0.1, float(table[versions].to_numpy(dtype=float).max()) * 1.18))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel(xlabel)
    handles = _patches(VERSION_COLORS, versions)
    if reference is not None:
        ax.axvline(reference[1], color=INK_3, linewidth=1.2, linestyle=(0, (3, 3)))
        handles.append(mpl.lines.Line2D([], [], color=INK_3, linewidth=1.2, linestyle=(0, (3, 3)),
                                        label=f"{reference[0]} ({reference[1]:.0%})"))
    _legend_top(ax, handles)
    ax.set_title(title, pad=24)
    return fig


def plot_world_probe(probe: pd.DataFrame) -> plt.Figure:
    """Text-only world probe accuracy per text version, against its two chance levels."""
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(probe) + 1.4))
    y = np.arange(len(probe))
    acc = probe["world_accuracy"].astype(float)
    ax.barh(y, acc, color=[VERSION_COLORS.get(i, SERIES[0]) for i in probe.index], height=0.55)
    for yi, v in zip(y, acc):
        ax.text(v + 0.002, yi, f"{v:.1%}", va="center", fontsize=8.5, color=INK_2)
    for col, label, style in [("chance_given_category", "chance, same domain × category", (0, (3, 3))),
                              ("chance_given_domain", "chance, same domain", (0, (1, 2)))]:
        c = float(probe[col].iloc[0])
        ax.axvline(c, color=INK_3, linewidth=1.2, linestyle=style, label=f"{label} ({c:.1%})")
    ax.legend(loc="lower right", handlelength=2.2)
    ax.set_yticks(y, probe.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(0.05, float(acc.max()) * 1.25))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("val trajectories whose nearest train neighbour (TF-IDF) has the same world")
    ax.set_title("Can the text alone identify the simulated world?")
    return fig


def plot_provenance_before_after(table: pd.DataFrame) -> plt.Figure:
    """Heatmaps side by side: provenance rates on each text version present (raw, anonymised, masked)."""
    versions = [v for v in VERSION_COLORS if v in set(table.columns.get_level_values(1))]
    tabs = {v: table.xs(v, axis=1, level=1) for v in versions}
    vmax = max(float(t.to_numpy().max()) for t in tabs.values())
    fig, axes = plt.subplots(1, len(versions), figsize=(6 * len(versions), 0.42 * len(tabs[versions[0]]) + 1.6),
                             sharey=True)
    for ax, v in zip(np.atleast_1d(axes), versions):
        heatmap(ax, tabs[v], vmax=vmax)
        ax.set_title(f"{v} text", fontsize=10)
    fig.suptitle("Argument provenance on each version of the text (share of steps)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_injection_signals(table: pd.DataFrame) -> plt.Figure:
    """Per attack type, share of poisoned responses carrying each signal, before / after masking."""
    signals = list(dict.fromkeys(table.columns.get_level_values(0)))
    versions = [v for v in VERSION_COLORS if v in set(table.columns.get_level_values(1))]
    fig, axes = plt.subplots(1, len(signals), figsize=(4.2 * len(signals), 0.55 * len(table) + 1.6), sharey=True)
    h = 0.8 / len(versions)
    y = np.arange(len(table))
    for ax, sig in zip(np.atleast_1d(axes), signals):
        for k, v in enumerate(versions):
            vals = table[(sig, v)].astype(float).to_numpy()
            yy = y + (k - (len(versions) - 1) / 2) * h
            ax.barh(yy, vals, height=h * 0.9, color=VERSION_COLORS[v])
            for yi, val in zip(yy, vals):
                ax.text(val + 0.02, yi, "0" if val == 0 else ("<1%" if val < 0.005 else f"{val:.0%}"),
                        va="center", fontsize=7.5, color=INK_2)
        ax.set_title(textwrap.fill(sig, 26), fontsize=9.5)
        ax.set_xlim(0, 1.25)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        ax.grid(axis="y", visible=False)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(y, [f"{i} injections" for i in table.index])
    ax0.invert_yaxis()
    _figure_header(fig, "What is left in the poisoned response once the generator's tics are masked?",
                   _patches(VERSION_COLORS, versions), top=0.82)
    return fig


# --------------------------------------------------------------------------- part C: evaluation protocol


SPLIT_COLORS = {"template folds": SERIES[0], "world folds": SERIES[1]}
TARGET_NAMES = {"y_traj": "y_traj (trajectory)", "y_gate": "y_gate (step)", "y_scan": "y_scan (step)"}


def plot_fold_prevalence(fp: pd.DataFrame) -> plt.Figure:
    """Prevalence per dev fold, template folds (protocol) against world folds (control), one panel per target."""
    targets = [t for t in TARGET_NAMES if t in set(fp["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(11, 2.8))
    for ax, t in zip(np.atleast_1d(axes), targets):
        sub = fp[fp["target"] == t]
        for k, (split, color) in enumerate(SPLIT_COLORS.items()):
            s = sub[sub["split"] == split].sort_values("fold")
            ax.plot(s["fold"] + (k - 0.5) * 0.12, s["prevalence"], "o", color=color, markersize=6, label=split)
        ax.set_title(TARGET_NAMES[t], fontsize=10)
        ax.set_xticks(sorted(sub["fold"].unique()))
        ax.set_xlabel("fold")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        lo, hi = sub["prevalence"].min(), sub["prevalence"].max()
        pad = max(0.01, (hi - lo) * 0.25)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("prevalence")
    _figure_header(fig, "Prevalence per fold: grouping by template keeps folds balanced, grouping by world does not",
                   _patches(SPLIT_COLORS, SPLIT_COLORS))
    return fig


def _figure_header(fig: plt.Figure, title: str, handles: list, top: float = 0.80) -> None:
    """Bold title and a one-row legend above a multi-panel figure, without the gap tight_layout leaves."""
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.suptitle(title, x=0.01, y=0.99, ha="left", va="top", fontsize=11, fontweight="bold")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99 - 0.09 * 2.8 / fig.get_figheight()),
               ncol=len(handles), handlelength=1.0, handleheight=0.8, columnspacing=1.2, borderaxespad=0)


def plot_resolution(res: pd.DataFrame, fpr_name: str, min_fp: int) -> plt.Figure:
    """Expected false positives at one target FPR, per evaluation unit, for the gate and the scanner."""
    views = {"gate": SERIES[0], "scan": SERIES[2]}
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(res) + 1.4))
    y = np.arange(len(res))
    h = 0.38
    for k, (v, color) in enumerate(views.items()):
        vals = res[f"{v} FP @ {fpr_name}"].astype(float).to_numpy()
        yy = y + (k - 0.5) * h
        ax.barh(yy, vals, height=h * 0.9, color=color, label=v)
        for yi, val in zip(yy, vals):
            ax.text(val + 0.3, yi, f"{val:.0f}", va="center", fontsize=7.5, color=INK_2)
    ax.axvline(min_fp, color=INK_3, linewidth=1.2, linestyle=(0, (3, 3)))
    ax.text(min_fp + 0.3, -0.9, f"{min_fp} false positives: relative SE of the FPR ≈ {1 / np.sqrt(min_fp):.0%}",
            fontsize=8, color=INK_3, va="center")
    ax.set_yticks(y, res.index)
    ax.invert_yaxis()
    ax.set_ylim(len(res) - 0.4, -1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"expected false positives = negative steps × {fpr_name}")
    _legend_top(ax, _patches(views, views), y=1.01)
    ax.set_title(f"How many false positives does a {fpr_name} FPR leave to measure?", pad=22)
    return fig


WIDTH_COLORS = {"width (fold, steps iid)": INK_3, "width (fold, templates)": SERIES[0],
                "width (test size, templates)": SERIES[6]}
WIDTH_LABELS = {"width (fold, steps iid)": "steps resampled (invalid)",
                "width (fold, templates)": "templates resampled, fold size",
                "width (test size, templates)": "templates resampled, test size"}


def plot_interval_widths(summaries: dict, metrics: list[str]) -> plt.Figure:
    """95% interval width of the reference score per metric: step bootstrap vs template bootstrap."""
    fig, axes = plt.subplots(1, len(summaries), figsize=(11, 0.55 * len(metrics) + 1.6), sharey=True)
    cols = list(WIDTH_COLORS)
    h = 0.8 / len(cols)
    for ax, (target, s) in zip(np.atleast_1d(axes), summaries.items()):
        s = s.loc[metrics]
        y = np.arange(len(s))
        for k, c in enumerate(cols):
            yy = y + (k - (len(cols) - 1) / 2) * h
            ax.barh(yy, s[c], height=h * 0.9, color=WIDTH_COLORS[c])
            for yi, v in zip(yy, s[c]):
                ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(y, metrics)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, float(s[cols].to_numpy().max()) * 1.25)
        ax.set_title(TARGET_NAMES.get(target, target), fontsize=10)
        ax.set_xlabel("width of the 95% interval")
    np.atleast_1d(axes)[0].invert_yaxis()  # shared y axis: invert once
    leg = [Patch(facecolor=WIDTH_COLORS[c], label=WIDTH_LABELS[c]) for c in cols]
    _figure_header(fig, "Expected 95% interval widths (metadata-only reference score, dev folds)", leg)
    return fig
