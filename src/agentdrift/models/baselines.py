"""v0.0 trivial and shortcut baselines, v0.1 TF-IDF + logistic regression (KAN-20).

Each function is an estimator FACTORY for the experiment harness (`models/experiment.py`): it returns a fresh,
unfitted scikit-learn estimator that takes the view's feature DataFrame as X. A configuration names it in
`configs/models/<version>.yaml` as `model.factory: agentdrift.models.baselines:<function>`.

So far (KAN-61): `majority`, the harness's end-to-end check. The other v0.0 / v0.1 baselines come with KAN-20.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier


def majority() -> DummyClassifier:
    """Constant score = training prevalence (`strategy="prior"`): ROC-AUC 0.5, PR-AUC = prevalence, no alert at
    any FPR target. Reads no feature; it only checks the harness end to end."""
    return DummyClassifier(strategy="prior")
