"""Text block (B.2, KAN-17): TF-IDF per field of a view, fitted inside the training fold only (L2).

    from agentdrift.features.text import TextBlock, text_block
    block = TextBlock(view="gate", analyzer="word")          # a scikit-learn transformer: fit on the training rows
    X_train = block.fit_transform(view_df_train[features])    # one vectoriser per field, vocabulary of these rows only

In a model version, through a factory of `models/baselines.py` (`model.params.text: {analyzer: char, ...}`).

Design (PLAN_FEATURES step 1):

    fields      one vectoriser PER FIELD of the view: the same word does not mean the same thing in the task, the
                history, the tool response or the proposed call. Feature names are `<field>__<kind>__<token>`.
                `fields=None` takes every column of X at fit time: the harness gives only the view's feature columns
                (`columns:` of the YAML, checked by `check_columns`), so the config stays the single source of truth
                (e.g. the gate "action only" variant). Any column that is not a feature of the view is refused.
    analyzer    "word" (1-2 grams), "char" (`char_wb` 3-5 grams) or "both" (two vectorisers per field).
    placeholders  anonymisation / masking placeholders (`<EXT_EMAIL_3>`, `<CONTACT_2_EMAIL>`, `<AMOUNT_1>`, `<HEADER>`),
                history tags (`<call>`, `<result>`, `<thought>`) and markers (`<history truncated>`) stay ATOMIC: one
                word token each. For the character analyser each is replaced by private-use characters that never
                occur in the data: one fixed code per placeholder TYPE (`KNOWN_TYPES`, collision-free) followed, for
                indexed ones, by one code for the index; so no n-gram cuts a placeholder into pieces shared with
                ordinary text (`<EXT_` would otherwise be a feature of its own). An unknown type gets a crc32 code in a
                separate range (a collision there is possible, and only between unknown types).
                `placeholder_index=False` drops the causal index (`<EXT_EMAIL_3>` -> `<EXT_EMAIL>`): the index says how
                many distinct entities appeared before, an ablation, not the default.
    no cleaning JSON arguments are read as text (keys and values); no stop words, no stripping: an injection cue
                must not be cleaned away. `lowercase=True` by default (casing is part of the generator's style).
    weights     per-field multipliers (`{"history_text": 0.5}`), applied after the TF-IDF normalisation.
    svd         `text_block(..., svd=k)` appends a TruncatedSVD for dense models (k-NN of the AutoML).

Missing text (none in AgentDrift; possible in ATBench) is read as an empty string. A field with no token at all in the
training rows (e.g. arguments that are always `{}`) gives an empty block (0 columns, listed in `empty_fields_`), not
an error: its absence of signal is a fact of these rows. Nothing here reads a label.
"""

from __future__ import annotations

import logging
import re
import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from agentdrift.features.views import VIEWS, feature_columns

log = logging.getLogger(__name__)

ANALYZERS = ("word", "char", "both")
# Anonymisation / masking placeholders (upper case, digits, underscores) and history tags / markers (lower case,
# at most one inner space: `<history truncated>`, `<missing steps>`). Ordinary markup such as `<a href=...>` is not one.
PLACEHOLDER = re.compile(r"<(?:[A-Z][A-Z0-9_]*|[a-z]+(?: [a-z]+)?)>")
_INDEX = re.compile(r"_(\d+)(?=_|>)")
# Word tokens: a placeholder as one token, else sklearn's default (2+ word characters). Applied after lowercasing.
WORD_TOKEN_PATTERN = r"<(?:[a-zA-Z][a-zA-Z0-9_]*|[a-z]+(?: [a-z]+)?)>|\b\w\w+\b"
# Every placeholder type written by `data/anonymize.py`, `features/masking.py` and `features/views.py` (index dropped).
# Order is part of the encoding: append, never reorder.
KNOWN_TYPES = (
    "<USER>", "<USER_EMAIL>", "<COMPANY>", "<CONTACT>", "<CONTACT_EMAIL>", "<INT_EMAIL>", "<EXT_EMAIL>", "<INT_URL>",
    "<EXT_URL>", "<INT_DOMAIN>", "<EXT_DOMAIN>", "<AMOUNT>", "<HEADER>", "<CODE>",
    "<call>", "<result>", "<thought>", "<history truncated>", "<missing steps>",
)
# Unicode private-use areas, never in the data: known types, unknown types (crc32), indices.
_KNOWN_START, _UNKNOWN_START, _UNKNOWN_SIZE, _INDEX_START = 0xE000, 0xE100, 0x1800, 0xF0000


def drop_index(placeholder: str) -> str:
    """`<EXT_EMAIL_3>` -> `<EXT_EMAIL>`, `<CONTACT_2_EMAIL>` -> `<CONTACT_EMAIL>`; tags unchanged."""
    return _INDEX.sub("", placeholder)


def atomic_chars(placeholder: str, keep_index: bool = True) -> str:
    """Private-use code of a placeholder: one character for its type (+ one for its index). Fixed, no fitting."""
    kind = drop_index(placeholder)
    code = (chr(_KNOWN_START + KNOWN_TYPES.index(kind)) if kind in KNOWN_TYPES
            else chr(_UNKNOWN_START + zlib.crc32(kind.encode()) % _UNKNOWN_SIZE))
    if keep_index:
        code += "".join(chr(_INDEX_START + int(k)) for k in _INDEX.findall(placeholder))
    return code


class Preprocessor:
    """Callable given to TfidfVectorizer as `preprocessor` (picklable, compared by value for caching)."""

    def __init__(self, kind: str, lowercase: bool = True, placeholder_index: bool = True):
        self.kind, self.lowercase, self.placeholder_index = kind, lowercase, placeholder_index

    def _placeholder(self, m: re.Match) -> str:
        if self.kind == "char":
            return atomic_chars(m.group(0), self.placeholder_index)
        p = m.group(0) if self.placeholder_index else drop_index(m.group(0))
        return p.lower() if self.lowercase else p

    def __call__(self, text: str) -> str:
        parts, last = [], 0
        for m in PLACEHOLDER.finditer(text):  # lowercase the text between placeholders, never inside them
            seg = text[last:m.start()]
            parts += [seg.lower() if self.lowercase else seg, self._placeholder(m)]
            last = m.end()
        seg = text[last:]
        parts.append(seg.lower() if self.lowercase else seg)
        return "".join(parts)

    def __eq__(self, other) -> bool:
        return isinstance(other, Preprocessor) and vars(self) == vars(other)

    def __hash__(self) -> int:
        return hash(tuple(sorted(vars(self).items())))

    def __repr__(self) -> str:
        return f"Preprocessor({self.kind!r}, lowercase={self.lowercase}, placeholder_index={self.placeholder_index})"


class TextBlock(TransformerMixin, BaseEstimator):
    """Per-field TF-IDF of a view (see the module docstring). Sparse float32 output."""

    def __init__(self, view: str | None = None, fields: list[str] | None = None, analyzer: str = "word",
                 word_ngram: tuple[int, int] = (1, 2), char_ngram: tuple[int, int] = (3, 5), min_df: int | float = 2,
                 max_df: float = 1.0, max_features: int | None = None, sublinear_tf: bool = True,
                 lowercase: bool = True, placeholder_index: bool = True, weights: dict[str, float] | None = None):
        self.view = view
        self.fields = fields
        self.analyzer = analyzer
        self.word_ngram = word_ngram
        self.char_ngram = char_ngram
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.lowercase = lowercase
        self.placeholder_index = placeholder_index
        self.weights = weights

    # -------------------------------------------------------------------- checks

    def _fields(self, X: pd.DataFrame) -> list[str]:
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {self.view!r}")
        if self.analyzer not in ANALYZERS:
            raise ValueError(f"analyzer must be one of {ANALYZERS}, got {self.analyzer!r}")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TextBlock reads named fields: X must be a DataFrame of view columns")
        allowed = set(feature_columns(self.view, include_thought=True))  # type: ignore[arg-type]
        not_features = sorted(set(X.columns) - allowed)
        if not_features:  # never read keys, labels, groups, diagnostics or metadata (L3, L12)
            raise ValueError(f"columns {not_features} are not text features of the {self.view} view")
        fields = list(self.fields) if self.fields is not None else list(X.columns)
        missing = sorted(set(fields) - set(X.columns))
        if missing:
            raise ValueError(f"fields {missing} are not in X")
        if not fields:
            raise ValueError("no text field to vectorise")
        unknown_w = sorted(set(self.weights or {}) - set(fields))
        if unknown_w:
            raise ValueError(f"weights given for fields that are not vectorised: {unknown_w}")
        return fields

    def _vectorizer(self, kind: str) -> TfidfVectorizer:
        common = {"min_df": self.min_df, "max_df": self.max_df, "max_features": self.max_features,
                  "sublinear_tf": self.sublinear_tf, "lowercase": False, "dtype": np.float32,
                  "preprocessor": Preprocessor(kind, self.lowercase, self.placeholder_index)}
        if kind == "word":
            return TfidfVectorizer(analyzer="word", ngram_range=tuple(self.word_ngram),
                                   token_pattern=WORD_TOKEN_PATTERN, **common)
        return TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(self.char_ngram), **common)

    @staticmethod
    def _column(X: pd.DataFrame, field: str) -> pd.Series:
        return X[field].fillna("").astype(str)

    # -------------------------------------------------------------------- sklearn API

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        fields = self._fields(X)
        kinds = ("word", "char") if self.analyzer == "both" else (self.analyzer,)
        self.vectorizers_, self.empty_fields_, blocks = [], [], []
        for field in fields:
            for kind in kinds:
                vec = self._vectorizer(kind)
                try:
                    block = vec.fit_transform(self._column(X, field))
                except ValueError as err:  # "empty vocabulary" / "After pruning, no terms remain"
                    if "vocabulary" not in str(err) and "no terms remain" not in str(err):
                        raise
                    log.warning("field %s (%s) has no token in these training rows: empty block", field, kind)
                    self.empty_fields_.append(f"{field}__{kind}")
                    blocks.append(sp.csr_matrix((len(X), 0), dtype=np.float32))
                    self.vectorizers_.append((field, kind, None))
                    continue
                blocks.append(block * float((self.weights or {}).get(field, 1.0)))
                self.vectorizers_.append((field, kind, vec))
        self.fields_ = fields
        self.n_features_out_ = int(sum(b.shape[1] for b in blocks))
        return sp.hstack(blocks, format="csr", dtype=np.float32)

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "vectorizers_")
        missing = sorted(set(self.fields_) - set(X.columns))
        if missing:
            raise ValueError(f"fields {missing} seen at fit time are missing from X")
        blocks = [sp.csr_matrix((len(X), 0), dtype=np.float32) if vec is None
                  else vec.transform(self._column(X, field)) * float((self.weights or {}).get(field, 1.0))
                  for field, _, vec in self.vectorizers_]
        return sp.hstack(blocks, format="csr", dtype=np.float32)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "vectorizers_")
        return np.concatenate([[f"{field}__{kind}__{t}" for t in vec.get_feature_names_out()] if vec is not None
                               else np.array([], dtype=object) for field, kind, vec in self.vectorizers_])

    def vocabulary_sizes(self) -> dict[str, int]:
        """Dimension of each (field, analyser) block, for the JOURNAL and the report."""
        check_is_fitted(self, "vectorizers_")
        return {f"{field}__{kind}": 0 if vec is None else len(vec.vocabulary_) for field, kind, vec in self.vectorizers_}


def text_block(view: str, svd: int | None = None, seed: int = 42, **params) -> TextBlock | Pipeline:
    """The text block, followed by a TruncatedSVD of `svd` components for dense models (fitted in the fold too)."""
    block = TextBlock(view=view, **params)
    if svd is None:
        return block
    return Pipeline([("tfidf", block), ("svd", TruncatedSVD(n_components=svd, random_state=seed))])
