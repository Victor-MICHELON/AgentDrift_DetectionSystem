"""Helpers for the presentation notebooks (EDA parts A, B, C).

The explanations of the notebooks are Markdown cells, so the numbers they quote are typed. To keep the notebooks
reproducible, each one ends with a `check` cell: every quoted number is recomputed, formatted the same way and
compared with the text. A mismatch raises, so "Restart & Run All" fails as soon as the text is out of date.
"""

from __future__ import annotations

import pandas as pd
from IPython.display import display

from agentdrift.viz import plots as P


def pct(x: float, digits: int = 0) -> str:
    """0.174 -> '17%' (digits=0) or '17.4%' (digits=1): the format used in the Markdown text."""
    return f"{100 * float(x):.{digits}f}%"


def num(x: float, digits: int = 0) -> str:
    """12536 -> '12,536'; 0.8764 -> '0.876' (digits=3)."""
    return f"{float(x):,.{digits}f}"


def show(fig, name: str) -> None:
    """Save the figure (results/figures PNG + report PDF) and display it once."""
    P.save(fig, name)
    display(fig)
    P.plt.close(fig)


def table(df: pd.DataFrame, pct_cols=(), pct_digits: int = 1, int_cols=(), dec_cols: dict | None = None,
          index: bool = True, caption: str | None = None):
    """Styled table: percentages, thousands separators and decimals by column; optional caption."""
    fmt = {c: (lambda v, d=pct_digits: "" if pd.isna(v) else pct(v, d)) for c in pct_cols}
    fmt |= {c: (lambda v: "" if pd.isna(v) else num(v)) for c in int_cols}
    fmt |= {c: (lambda v, d=d: "" if pd.isna(v) else num(v, d)) for c, d in (dec_cols or {}).items()}
    s = df.style.format(fmt)
    if not index:
        s = s.hide(axis="index")
    if caption:
        s = s.set_caption(caption)
    return s.set_table_styles([{"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: 600;"}])


def check(quoted: list[tuple[str, object, str]]) -> None:
    """quoted = [(what, recomputed text, text quoted in the Markdown)]. Displays the comparison, raises on a mismatch."""
    rows = pd.DataFrame(quoted, columns=["number quoted in the text", "recomputed", "quoted"])
    rows["ok"] = rows["recomputed"].astype(str) == rows["quoted"].astype(str)
    display(rows.assign(ok=rows["ok"].map({True: "✓", False: "✗"})).style.hide(axis="index"))
    bad = rows[~rows["ok"]]
    if len(bad):
        raise AssertionError(f"{len(bad)} quoted number(s) out of date: {bad['number quoted in the text'].tolist()}")
    print(f"All {len(rows)} numbers quoted in the text match the data.")
