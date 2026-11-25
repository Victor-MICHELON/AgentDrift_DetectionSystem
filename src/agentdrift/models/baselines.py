"""v0.0 trivial and shortcut baselines, v0.1 TF-IDF + logistic regression, tuned and calibrated (KAN-20).

Each function is an estimator FACTORY for the experiment harness (`models/experiment.py`): it returns a fresh,
unfitted scikit-learn estimator that takes the view's feature DataFrame as X. A configuration names it in
`configs/models/<version>.yaml` as `model.factory: agentdrift.models.baselines:<function>`. Arguments `view`,
`memory` and `inner_cv` are filled by the harness when a factory declares them.

v0.0 (repères, never selected as a model; `kind: trivial` or `kind: shortcut` in their YAML):
    majority        constant score = training prevalence (harness check)
    random_scores   uniform random scores (control of the bootstrap: ROC-AUC 0.5 on average)
    position_only   empirical rate of the target at the step index t (`uses_position: true`): the floor announced by
                    the EDA (ROC-AUC 0.876 on y_scan, 0.779 on y_gate, train -> val)
    style_stats     10 surface statistics of the content just read (length, punctuation, casing, digits...) and a
                    gradient-boosted tree: measures how much of the score the FORM of the text gives (JOURNAL 25/11,
                    PR-AUC 0.855 on one scanner fold). A probe of the generator's register, not a model
    keyword_rule    count of generic prompt-injection phrases, fixed a priori, nothing fitted. It is NOT the
                    surface-feature baseline of the AgentDrift paper: the paper reports its results (F1 0.647, recall
                    0.880 / 0.082 / 0.231) but not its six features, so it cannot be rebuilt faithfully (KAN-20)

v0.1 (`tuned_text_logreg`, the official baseline of Step 2.3): text block + logistic regression, C chosen on the inner
splits, scores calibrated on the inner out-of-fold scores (see `TunedTextLogReg`).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from agentdrift.features.text import PLACEHOLDER, TextBlock

# Fields that hold the text the agent has just read (scanner) or is about to act on (gate). The history is left out
# of the v0.0 probes on purpose: its length grows with t, so its statistics would re-introduce the position shortcut.
CONTENT_FIELDS = {"scanner": ("current_content",), "gate": ("last_content", "proposed_call_args")}

# Generic prompt-injection phrasing (instruction override, urgency, secrecy, redirection), written from the
# vocabulary of public injection examples, not tuned on AgentDrift: a "rule" floor, declared as such.
INJECTION_PHRASES = (
    "ignore previous", "ignore all", "ignore the above", "disregard", "new instructions", "instead",
    "you must", "you are required", "immediately", "urgent", "do not tell", "do not inform", "without telling",
    "don't mention", "forward", "override", "system:", "important:", "action required", "verify your",
)

CALIBRATIONS = ("none", "sigmoid", "isotonic")


def _content(X: pd.DataFrame, view: str) -> pd.DataFrame:
    fields = [f for f in CONTENT_FIELDS[view] if f in X.columns]
    if not fields:
        raise ValueError(f"none of the content fields {CONTENT_FIELDS[view]} of the {view} view is in X")
    return X[fields].fillna("").astype(str)


# ----------------------------------------------------------------------------- v0.0


def majority() -> DummyClassifier:
    """Constant score = training prevalence (`strategy="prior"`): ROC-AUC 0.5, PR-AUC = prevalence, no alert at
    any FPR target. Reads no feature; it only checks the harness end to end."""
    return DummyClassifier(strategy="prior")


class RandomScores(ClassifierMixin, BaseEstimator):
    """Uniform scores, reproducible: the same rows in the same order get the same scores for a given seed."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        p = np.random.default_rng(self.seed + len(X)).random(len(X))
        return np.column_stack([1 - p, p])


def random_scores(seed: int = 42) -> RandomScores:
    return RandomScores(seed)


def position_only() -> Pipeline:
    """P(target | t) learned on the training rows: one indicator per step index, an almost unregularised logistic
    regression (C large) so the score is the empirical rate at t. Needs `columns: [step]` and `uses_position: true`;
    an index unseen in training gets the prevalence (all indicators 0 -> intercept)."""
    return Pipeline([
        ("t", ColumnTransformer([("step", OneHotEncoder(handle_unknown="ignore"), ["step"])])),
        ("clf", LogisticRegression(C=1e4, max_iter=1000)),
    ])


def surface_statistics(X: pd.DataFrame, view: str) -> pd.DataFrame:
    """10 form statistics per content field, no vocabulary: what a model can learn from the register alone."""
    out = {}
    for field, t in _content(X, view).items():
        n = t.str.len().clip(lower=1)
        out |= {
            f"{field}__log_length": np.log1p(t.str.len()),
            f"{field}__words": t.str.split().str.len(),
            f"{field}__sentences": t.str.count(r"[.!?](?:\s|$)"),
            f"{field}__upper_ratio": t.str.count(r"[A-Z]") / n,
            f"{field}__digit_ratio": t.str.count(r"\d") / n,
            f"{field}__punct_ratio": t.str.count(r"[^\w\s]") / n,
            f"{field}__json": t.str.lstrip().str.startswith(("{", "[")).astype(int),
            f"{field}__colons": t.str.count(":"),
            f"{field}__newlines": t.str.count(r"\\n|\n"),
            f"{field}__placeholders": t.map(lambda s: len(PLACEHOLDER.findall(s))),
        }
    return pd.DataFrame(out, index=X.index)


def style_stats(view: str, seed: int = 42) -> Pipeline:
    """Form statistics (`surface_statistics`) + HistGradientBoostingClassifier with fixed default-like parameters
    (no tuning: a probe, not a model). Boosting is used here only because the form signal is non-linear (lengths,
    ratios); boosting as a MODEL is v0.3 (KAN-23)."""
    return Pipeline([
        ("stats", FunctionTransformer(surface_statistics, kw_args={"view": view})),
        ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)),
    ])


class KeywordRule(ClassifierMixin, BaseEstimator):
    """Score = number of distinct `INJECTION_PHRASES` found (lowercased) in the content fields, divided by the number
    of phrases. Nothing is fitted; `fit` only records the classes."""

    def __init__(self, view: str | None = None, phrases: tuple[str, ...] = INJECTION_PHRASES):
        self.view, self.phrases = view, phrases

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        text = _content(X, self.view).agg(" ".join, axis=1).str.lower()
        hits = sum(text.str.contains(re.escape(p), regex=True).astype(int) for p in self.phrases)
        p = np.asarray(hits, dtype=float) / len(self.phrases)
        return np.column_stack([1 - p, p])


def keyword_rule(view: str) -> KeywordRule:
    return KeywordRule(view)


# ----------------------------------------------------------------------------- v0.1


def tfidf_logreg(view: str, memory: str | None = None, C: float = 1.0, class_weight: str | None = "balanced",
                 max_iter: int = 2000, text: dict | None = None) -> Pipeline:
    """Text block (`features/text.TextBlock`, parameters in `text`) + logistic regression, refitted in every fold.

    `view` and `memory` are filled by the harness (memory: the run's cache, so an identical TF-IDF fit on the same rows
    is computed once). `class_weight="balanced"` reweights the minority class: the scores rank well but are not
    probabilities. No tuning, no calibration: the official v0.1 is `tuned_text_logreg`."""
    return Pipeline([("text", TextBlock(view=view, **(text or {}))),
                     ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter))], memory=memory)


class TunedTextLogReg(ClassifierMixin, BaseEstimator):
    """v0.1: text block + logistic regression, C tuned and the scores calibrated, all on the INNER splits.

    fit(X, y) on the training part of an outer split, with `inner_cv` its inner splits (grouped by template):
      1. for each C of `Cs` and each inner split: fit on the inner training rows, score the inner validation rows
         (decision function). C* = the best mean PR-AUC over the inner splits (ties: the smallest C, the most
         regularised).
      2. the inner out-of-fold scores of C* fit the calibration map (`calibration`):
            none      the logistic regression's own probability, sigmoid(decision). With class_weight="balanced"
                      the positives weigh as much as the negatives in the loss: the probabilities are shifted up
                      (the model behaves as if the prevalence were 50 %), they rank well but are not frequencies.
            sigmoid   Platt scaling: a logistic regression of y on the decision score (unregularised). Strictly
                      increasing: the ranking, hence PR-AUC, ROC-AUC and recall at a fixed FPR, is unchanged.
            isotonic  isotonic regression: non-decreasing step function, more flexible, but it merges scores into
                      plateaus (ties), so ranking metrics CAN change, and it may overfit with few positives.
         This is scikit-learn's CalibratedClassifierCV(ensemble=False) recipe: calibrator fitted on cross-validated
         predictions, then applied to the model refitted on all the training rows.
      3. `inner_oof_scores_` = the calibrated inner out-of-fold scores: the harness sets the FPR thresholds on them
         (`threshold_source: estimator`), on the same scale as the final scores. They come from the C chosen on the
         same splits, so they are slightly optimistic (JOURNAL 25/11: compared once with `inner_oof`).
      4. the final model = C* refitted on all the training rows.
    `selected_` records C* and the inner PR-AUC of every C (written in metrics.json by the harness).
    The vectoriser is the same for every C: with `memory` (the harness cache) each inner split is vectorised once.
    """

    def __init__(self, view: str | None = None, memory: str | None = None, inner_cv=None,
                 Cs: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0), class_weight: str | None = "balanced",
                 calibration: str = "sigmoid", max_iter: int = 2000, text: dict | None = None):
        self.view = view
        self.memory = memory
        self.inner_cv = inner_cv
        self.Cs = Cs
        self.class_weight = class_weight
        self.calibration = calibration
        self.max_iter = max_iter
        self.text = text

    def _pipeline(self, C: float) -> Pipeline:
        return tfidf_logreg(self.view, self.memory, C, self.class_weight, self.max_iter, self.text)

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calibration == "none":
            return 1 / (1 + np.exp(-raw))
        if self.calibration == "sigmoid":
            return self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
        return self.calibrator_.predict(raw)

    def fit(self, X: pd.DataFrame, y):
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {CALIBRATIONS}")
        if not self.inner_cv:
            raise ValueError("TunedTextLogReg needs the inner splits of its training rows (`inner_cv`)")
        y = np.asarray(y).astype(int)
        oof = {C: np.full(len(y), np.nan) for C in self.Cs}
        ap = {C: [] for C in self.Cs}
        for itr, iva in self.inner_cv:
            for C in self.Cs:
                s = self._pipeline(C).fit(X.iloc[itr], y[itr]).decision_function(X.iloc[iva])
                oof[C][iva] = s
                ap[C].append(average_precision_score(y[iva], s))
        mean_ap = {C: float(np.mean(v)) for C, v in ap.items()}
        best = min(self.Cs, key=lambda C: (-round(mean_ap[C], 6), C))
        raw = oof[best]
        if np.isnan(raw).any():
            raise RuntimeError("inner splits did not cover the training rows")
        if self.calibration == "sigmoid":
            self.calibrator_ = LogisticRegression(C=np.inf).fit(raw.reshape(-1, 1), y)
        elif self.calibration == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, y)
        self.model_ = self._pipeline(best).fit(X, y)
        self.inner_oof_scores_ = self._calibrate(raw)
        self.selected_ = {"C": best, "inner_pr_auc": {str(C): v for C, v in mean_ap.items()}}
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

    def predict_proba(self, X) -> np.ndarray:
        p = self._calibrate(self.decision_function(X))
        return np.column_stack([1 - p, p])


def tuned_text_logreg(view: str, inner_cv, memory: str | None = None, Cs: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0),
                      class_weight: str | None = "balanced", calibration: str = "sigmoid", max_iter: int = 2000,
                      text: dict | None = None) -> TunedTextLogReg:
    return TunedTextLogReg(view, memory, inner_cv, tuple(Cs), class_weight, calibration, max_iter, text)


def top_coefficients(model: TunedTextLogReg | Pipeline, k: int = 30) -> pd.DataFrame:
    """The k most positive and k most negative coefficients of each field (`<field>__<kind>__<token>`), to be read
    BY HAND for residual shortcuts (position tokens, generator tics, suspicious placeholders)."""
    pipe = model.model_ if isinstance(model, TunedTextLogReg) else model
    names = pipe.named_steps["text"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coef": coef})
    parts = df["feature"].str.split("__", n=2, expand=True)
    df["field"], df["kind"], df["token"] = parts[0], parts[1], parts[2]
    pos = df.sort_values("coef", ascending=False).groupby("field").head(k).assign(sign="+")
    neg = df.sort_values("coef").groupby("field").head(k).assign(sign="-")
    return pd.concat([pos, neg]).sort_values(["field", "sign", "coef"], ascending=[True, True, False])[
        ["field", "sign", "token", "coef", "kind"]].reset_index(drop=True)
