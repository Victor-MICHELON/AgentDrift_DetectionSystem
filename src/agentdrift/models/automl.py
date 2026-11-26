"""AutoML v0.2 (B.6, KAN-21): preparation, feature engineering and selection, model selection and hyperparameters,
searched by Optuna on the INNER splits of each outer training part (nested cross-validation).

    model.factory: agentdrift.models.automl:automl        (configs/models/v0_2_automl.yaml)

The four bricks of Step 2.5, all inside ONE scikit-learn Pipeline fitted on the rows it is given:

    preparation       text: missing text read as "" (TextBlock); non-text blocks: SimpleImputer(0) then MaxAbsScaler
                      (no missing value in AgentDrift; kept for ATBench and truncated views); categories (risk
                      category, transitions) are indicators built by the blocks
    features          blocks (a hyperparameter, per view, from the H1 result: structure only helped the gate),
                      TF-IDF configuration (a hyperparameter), selection SelectKBest(chi2) of k features (a
                      hyperparameter; chi2 because every block is non-negative; L1 selection was left out: one more
                      model fit per trial for a selection that the L2 logistic regression already performs implicitly)
    models            logreg, linear_svm (LinearSVC, its decision function as score; calibrated afterwards like the
                      others), complement_nb (on TF-IDF counts-like features), decision_tree (on the top-k chi2
                      features), forest (random forest or extra trees, on a TruncatedSVD), knn (on a TruncatedSVD,
                      L2-normalised). No boosting (v0.3, KAN-23). No RBF SVM: kernel cost grows quadratically with
                      ~35 k inner training rows, for a problem a linear model already separates at PR-AUC ~0.97
    hyperparameters   Optuna 5.1.0, TPE sampler with a fixed seed, objective = mean PR-AUC over the inner splits,
                      median pruner after each inner split (from the 9th trial). A FIXED number of trials (the
                      search is reproducible); `timeout` is only a safety cap on ACTIVE compute time
                      (`time.perf_counter`, which stops while the machine sleeps), checked after every trial; whether
                      it was hit is recorded. Optuna's own `timeout` is NOT used: it counts wall-clock time, so a laptop
                      going to sleep consumed the budget and cut a search to 3 trials (JOURNAL 26/11)

`fit(X, y)` on the training part of an outer split, `inner_cv` its inner splits (grouped by template):
  1. Optuna search: every trial builds a pipeline, fits it on each inner training part, scores the inner validation
     part. Expensive steps take DISCRETE values (TF-IDF configuration, blocks, SVD size) and the pipeline caches
     them, so a vectorisation is computed once per inner split and configuration. The cache is PRIVATE to one `fit`
     (a temporary directory deleted at its end): cached feature matrices take 50-100 MB each, and keeping them across
     the outer splits would grow to tens of GB.
  2. the best trial's inner out-of-fold scores fit a Platt calibration (as v0.1); calibrated, they are
     `inner_oof_scores_`, from which the harness sets the FPR thresholds (`threshold_source: estimator`).
  3. the best pipeline is refitted on all the rows given.
`selected_` (written per fold in metrics.json): family, all parameters, inner PR-AUC, trials (complete / pruned),
whether the timeout was hit, and the best trial of every family.
"""

from __future__ import annotations

import logging
import tempfile
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from agentdrift.features.provenance import InjectionCuesBlock, ProvenanceBlock
from agentdrift.features.structural import StructureBlock
from agentdrift.features.text import TextBlock

log = logging.getLogger(__name__)

FAMILIES = ("logreg", "linear_svm", "complement_nb", "decision_tree", "forest", "knn")
# Blocks offered per view (H1, JOURNAL 26/11: structure + provenance helped the gate out of domain, not the scanner).
BLOCK_SETS = {
    "scanner": {"text": ("text",), "text_cues": ("text", "cues"), "text_struct_cues": ("text", "structure", "cues")},
    "gate": {"text": ("text",), "text_struct": ("text", "structure", "provenance"),
             "text_struct_cues": ("text", "structure", "provenance", "cues")},
}
# TF-IDF configurations (discrete, so that the cache reuses them). Characters are 5x slower: opt-in (`allow_char`).
TEXT_CONFIGS = {
    "word12_min2": {"analyzer": "word", "word_ngram": (1, 2), "min_df": 2},
    "word12_min5": {"analyzer": "word", "word_ngram": (1, 2), "min_df": 5},
    "word1_min2": {"analyzer": "word", "word_ngram": (1, 1), "min_df": 2},
}
CHAR_CONFIGS = {"word_char": {"analyzer": "both", "min_df": 3, "max_features": 200_000}}


# ----------------------------------------------------------------------------- safe building blocks


class TopKChi2(TransformerMixin, BaseEstimator):
    """SelectKBest(chi2) with k clipped to the number of features (a small fold may have fewer than k)."""

    def __init__(self, k: int = 20_000):
        self.k = k

    def fit(self, X, y):
        self.selector_ = SelectKBest(chi2, k=min(self.k, X.shape[1])).fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "selector_")
        return self.selector_.transform(X)


class SafeSVD(TransformerMixin, BaseEstimator):
    """TruncatedSVD with n_components clipped below the number of features."""

    def __init__(self, n_components: int = 128, seed: int = 42):
        self.n_components, self.seed = n_components, seed

    def fit(self, X, y=None):
        self.svd_ = TruncatedSVD(n_components=max(1, min(self.n_components, X.shape[1] - 1)), random_state=self.seed)
        self.svd_.fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "svd_")
        return self.svd_.transform(X)


def _non_text(block) -> Pipeline:
    return Pipeline([("b", block), ("impute", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
                     ("scale", MaxAbsScaler())])


def features(view: str, blocks: tuple[str, ...], text: dict) -> FeatureUnion:
    make = {"text": lambda: TextBlock(view=view, **text), "structure": lambda: _non_text(StructureBlock(view)),
            "provenance": lambda: _non_text(ProvenanceBlock(view)), "cues": lambda: _non_text(InjectionCuesBlock(view))}
    return FeatureUnion([(b, make[b]()) for b in blocks])


def model_of(p: dict, seed: int):
    """The final estimator of a parameter set (keys as sampled by `sample`)."""
    fam = p["family"]
    if fam == "logreg":
        return LogisticRegression(C=p["lr_C"], class_weight="balanced", max_iter=3000)
    if fam == "linear_svm":
        return LinearSVC(C=p["svm_C"], class_weight="balanced", max_iter=5000, random_state=seed)
    if fam == "complement_nb":
        return ComplementNB(alpha=p["nb_alpha"])
    if fam == "decision_tree":
        return DecisionTreeClassifier(max_depth=p["tree_depth"], min_samples_leaf=p["tree_leaf"],
                                      class_weight="balanced", random_state=seed)
    if fam == "forest":
        cls = RandomForestClassifier if p["forest_kind"] == "random_forest" else ExtraTreesClassifier
        return cls(n_estimators=p["forest_trees"], max_depth=p["forest_depth"], min_samples_leaf=p["forest_leaf"],
                   max_features=p["forest_max_features"], class_weight="balanced_subsample", n_jobs=-1,
                   random_state=seed)
    if fam == "knn":
        return KNeighborsClassifier(n_neighbors=p["knn_k"], weights=p["knn_weights"], n_jobs=-1)
    raise ValueError(f"unknown family {fam!r}")


def pipeline_of(p: dict, view: str, memory: str | None, seed: int, text_configs: dict) -> Pipeline:
    """Preparation + features + selection / projection + model, as one Pipeline (fitted in the fold)."""
    steps = [("features", features(view, BLOCK_SETS[view][p["blocks"]], text_configs[p["text"]]))]
    fam = p["family"]
    if fam in ("forest", "knn"):
        steps.append(("svd", SafeSVD(p["svd"], seed)))
        if fam == "knn":
            steps.append(("norm", Normalizer()))
    elif fam == "decision_tree":
        steps.append(("select", TopKChi2(p["tree_k"])))
    elif p.get("select", "none") != "none":
        steps.append(("select", TopKChi2(int(p["select"]))))
    steps.append(("model", model_of(p, seed)))
    return Pipeline(steps, memory=memory)


def raw_score(pipe, X) -> np.ndarray:
    if hasattr(pipe, "decision_function") and hasattr(pipe[-1], "decision_function"):
        return np.asarray(pipe.decision_function(X), dtype=float)
    proba = pipe.predict_proba(X)
    return np.asarray(proba[:, list(pipe.classes_).index(1)], dtype=float)


def sample(trial: optuna.Trial, view: str, families: tuple[str, ...], text_configs: dict) -> dict:
    """The search space. Expensive choices are categorical (cache-friendly); model parameters are continuous."""
    p = {"family": trial.suggest_categorical("family", list(families)),
         "blocks": trial.suggest_categorical("blocks", list(BLOCK_SETS[view])),
         "text": trial.suggest_categorical("text", list(text_configs))}
    fam = p["family"]
    if fam in ("logreg", "linear_svm", "complement_nb"):
        p["select"] = trial.suggest_categorical("select", ["none", "20000", "50000"])
    if fam == "logreg":
        p["lr_C"] = trial.suggest_float("lr_C", 0.03, 100.0, log=True)
    elif fam == "linear_svm":
        p["svm_C"] = trial.suggest_float("svm_C", 0.003, 10.0, log=True)
    elif fam == "complement_nb":
        p["nb_alpha"] = trial.suggest_float("nb_alpha", 0.01, 3.0, log=True)
    elif fam == "decision_tree":
        p["tree_k"] = trial.suggest_categorical("tree_k", [2000, 5000])
        p["tree_depth"] = trial.suggest_int("tree_depth", 4, 30)
        p["tree_leaf"] = trial.suggest_int("tree_leaf", 1, 50, log=True)
    elif fam == "forest":
        p["svd"] = trial.suggest_categorical("svd", [64, 128])
        p["forest_kind"] = trial.suggest_categorical("forest_kind", ["random_forest", "extra_trees"])
        p["forest_trees"] = trial.suggest_categorical("forest_trees", [100, 300])
        p["forest_depth"] = trial.suggest_int("forest_depth", 6, 30)
        p["forest_leaf"] = trial.suggest_int("forest_leaf", 1, 20, log=True)
        p["forest_max_features"] = trial.suggest_categorical("forest_max_features", ["sqrt", "log2"])
    elif fam == "knn":
        p["svd"] = trial.suggest_categorical("svd", [64, 128])
        p["knn_k"] = trial.suggest_int("knn_k", 5, 100, log=True)
        p["knn_weights"] = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
    return p


# ----------------------------------------------------------------------------- the estimator


class AutoMLClassifier(ClassifierMixin, BaseEstimator):
    """Nested search on the inner splits, then refit of the best pipeline (see the module docstring)."""

    def __init__(self, view: str | None = None, inner_cv=None, n_trials: int = 50,
                 timeout: float | None = 1200.0, seed: int = 42, families: tuple[str, ...] = FAMILIES,
                 allow_char: bool = False, pruner_startup: int = 8):
        self.view = view
        self.inner_cv = inner_cv
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
        self.families = families
        self.allow_char = allow_char
        self.pruner_startup = pruner_startup

    def _text_configs(self) -> dict:
        return {**TEXT_CONFIGS, **(CHAR_CONFIGS if self.allow_char else {})}

    def fit(self, X: pd.DataFrame, y):
        with tempfile.TemporaryDirectory(prefix="agentdrift_automl_") as cache:
            self._fit(X, y, cache)
        self.model_.memory = None  # the cache directory no longer exists; prediction never uses it
        return self

    def _fit(self, X: pd.DataFrame, y, cache: str) -> None:
        if self.view not in BLOCK_SETS:
            raise ValueError(f"view must be one of {tuple(BLOCK_SETS)}")
        if not self.inner_cv:
            raise ValueError("AutoMLClassifier needs the inner splits of its training rows (`inner_cv`)")
        unknown = set(self.families) - set(FAMILIES)
        if unknown or not self.families:
            raise ValueError(f"families must be a non-empty subset of {FAMILIES}")
        y = np.asarray(y).astype(int)
        texts = self._text_configs()
        best: dict = {"value": -np.inf, "oof": None, "params": None}
        by_family: dict[str, dict] = {}
        start = time.perf_counter()

        def objective(trial: optuna.Trial) -> float:
            t0 = time.perf_counter()
            trial.set_user_attr("seconds", 0.0)
            p = sample(trial, self.view, tuple(self.families), texts)
            oof = np.full(len(y), np.nan)
            aps = []
            for k, (itr, iva) in enumerate(self.inner_cv):
                pipe = pipeline_of(p, self.view, cache, self.seed, texts)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipe.fit(X.iloc[itr], y[itr])
                oof[iva] = raw_score(pipe, X.iloc[iva])
                aps.append(average_precision_score(y[iva], oof[iva]))
                trial.set_user_attr("seconds", round(time.perf_counter() - t0, 2))
                trial.report(float(np.mean(aps)), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            value = float(np.mean(aps))
            fam = p["family"]
            if value > by_family.get(fam, {}).get("inner_pr_auc", -np.inf):
                by_family[fam] = {"inner_pr_auc": value, "params": p}
            if value > best["value"]:
                best.update(value=value, oof=oof, params=p)
            return value

        def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            log.info("automl %s trial %d %s %s/%s/%s value=%s %.1fs (active)", self.view, trial.number,
                     trial.state.name, trial.params.get("family"), trial.params.get("blocks"),
                     trial.params.get("text"), None if trial.value is None else round(trial.value, 4),
                     trial.user_attrs.get("seconds", float("nan")))
            if self.timeout is not None and time.perf_counter() - start > self.timeout:
                study.stop()  # active-time budget (perf_counter stops while the machine sleeps)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=self.pruner_startup, n_warmup_steps=0))
        study.optimize(objective, n_trials=self.n_trials, callbacks=[report])
        self.trials_ = [{"number": t.number, "state": t.state.name, "value": t.value,
                         "seconds": t.user_attrs.get("seconds"), **t.params} for t in study.trials]
        if best["oof"] is None:
            raise RuntimeError("no trial completed: raise the timeout or the number of trials")
        states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()

        raw = best["oof"]
        self.calibrator_ = LogisticRegression(C=np.inf).fit(raw.reshape(-1, 1), y)
        self.inner_oof_scores_ = self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
        self.model_ = pipeline_of(best["params"], self.view, cache, self.seed, texts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(X, y)
        self.params_ = best["params"]
        self.selected_ = {
            "family": best["params"]["family"], "blocks": best["params"]["blocks"], "text": best["params"]["text"],
            "params": best["params"], "inner_pr_auc": best["value"],
            "trials": {"requested": self.n_trials, "complete": int(states.get("COMPLETE", 0)),
                       "pruned": int(states.get("PRUNED", 0)), "failed": int(states.get("FAIL", 0))},
            "timeout_hit": len(study.trials) < self.n_trials,
            "search_seconds": round(time.perf_counter() - start, 1),
            "slowest_trials": sorted(({"number": t["number"], "family": t.get("family"), "seconds": t["seconds"]}
                                      for t in self.trials_ if t["seconds"] is not None),
                                     key=lambda t: -t["seconds"])[:3],
            "best_by_family": by_family,
        }
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        p = self.calibrator_.predict_proba(raw_score(self.model_, X).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])


def automl(view: str, inner_cv, n_trials: int = 50, timeout: float | None = 1200.0, seed: int = 42,
           families: tuple[str, ...] = FAMILIES, allow_char: bool = False, pruner_startup: int = 8) -> AutoMLClassifier:
    return AutoMLClassifier(view, inner_cv, n_trials, timeout, seed, tuple(families), allow_char, pruner_startup)
