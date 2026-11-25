"""v0.0 trivial and shortcut baselines, v0.1 TF-IDF + logistic regression (KAN-20).

Each function is an estimator FACTORY for the experiment harness (`models/experiment.py`): it returns a fresh,
unfitted scikit-learn estimator that takes the view's feature DataFrame as X. A configuration names it in
`configs/models/<version>.yaml` as `model.factory: agentdrift.models.baselines:<function>`.

So far: `majority` (KAN-61, the harness's end-to-end check) and `tfidf_logreg` (KAN-17, the text block behind a plain
logistic regression: no tuning, no calibration). The other v0.0 / v0.1 baselines come with KAN-20.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agentdrift.features.text import TextBlock


def majority() -> DummyClassifier:
    """Constant score = training prevalence (`strategy="prior"`): ROC-AUC 0.5, PR-AUC = prevalence, no alert at
    any FPR target. Reads no feature; it only checks the harness end to end."""
    return DummyClassifier(strategy="prior")


def tfidf_logreg(view: str, memory: str | None = None, C: float = 1.0, class_weight: str | None = "balanced",
                 max_iter: int = 2000, text: dict | None = None) -> Pipeline:
    """Text block (`features/text.TextBlock`, parameters in `text`) + logistic regression, refitted in every fold.

    `view` and `memory` are filled by the harness (memory: the run's cache, so an identical TF-IDF fit on the same rows
    is computed once). `class_weight="balanced"` reweights the minority class: the scores rank well but are not
    probabilities (calibration is v0.1's job, KAN-20)."""
    return Pipeline([("text", TextBlock(view=view, **(text or {}))),
                     ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter))], memory=memory)
