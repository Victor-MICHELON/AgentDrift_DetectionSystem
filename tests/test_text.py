"""Text block (features/text.py, KAN-17): fitted on the training rows only, reads view features only, keeps the
placeholders atomic, keeps the fields apart, deterministic, usable from a YAML through the harness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from agentdrift.features.text import KNOWN_TYPES, PLACEHOLDER, Preprocessor, TextBlock, atomic_chars, text_block
from agentdrift.features.views import feature_columns
from agentdrift.models import experiment as E
from agentdrift.models.baselines import tfidf_logreg

pytestmark = pytest.mark.filterwarnings("ignore:The least populated class:UserWarning")


def gate_rows(n: int = 6, extra: str = "") -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({"task": f"pay the invoice {i % 2}", "history_text": "<call> get_invoice {}\n<result> invoice ok",
                     "last_content": f"please send to <EXT_EMAIL_{i % 3 + 1}> now {extra}",
                     "proposed_call_tool": "send_email", "proposed_call_args": '{"to": "<CONTACT_1_EMAIL>"}'})
    return pd.DataFrame(rows)[feature_columns("gate")]


def test_vocabulary_is_learned_on_the_training_rows_only():
    train, other = gate_rows(), gate_rows(extra="zebracorn zebracorn")
    block = TextBlock(view="gate", min_df=1).fit(train)
    names = block.get_feature_names_out()
    assert not [n for n in names if "zebracorn" in n]
    np.testing.assert_allclose(block.transform(other).toarray(), block.transform(gate_rows()).toarray(), atol=1e-6)


@pytest.mark.parametrize("bad", ["label", "trajectory_id", "template_id", "step", "history_n_steps", "source_category"])
def test_non_feature_columns_are_refused(bad):
    X = gate_rows().assign(**{bad: "x"})
    with pytest.raises(ValueError, match="not text features"):
        TextBlock(view="gate").fit(X)


def test_explicit_fields_must_be_in_x_and_weights_must_name_them():
    with pytest.raises(ValueError, match="not in X"):
        TextBlock(view="gate", fields=["current_content"]).fit(gate_rows())
    with pytest.raises(ValueError, match="weights"):
        TextBlock(view="gate", fields=["task"], weights={"history_text": 2.0}).fit(gate_rows())
    with pytest.raises(ValueError, match="view"):
        TextBlock(view="portal").fit(gate_rows())


def test_scanner_fields_are_not_gate_fields():
    with pytest.raises(ValueError, match="not text features"):
        TextBlock(view="scanner").fit(gate_rows())  # proposed_call_* are not scanner features


@pytest.mark.parametrize("analyzer", ["word", "char", "both"])
def test_deterministic(analyzer):
    a = TextBlock(view="gate", analyzer=analyzer, min_df=1).fit_transform(gate_rows())
    b = clone(TextBlock(view="gate", analyzer=analyzer, min_df=1)).fit_transform(gate_rows())
    assert (a != b).nnz == 0 and a.dtype == np.float32


def test_fit_transform_equals_fit_then_transform():
    block = TextBlock(view="gate", analyzer="both", min_df=1)
    a = block.fit_transform(gate_rows())
    np.testing.assert_allclose(a.toarray(), block.transform(gate_rows()).toarray(), atol=1e-6)


def test_word_placeholders_are_single_tokens():
    names = TextBlock(view="gate", min_df=1, word_ngram=(1, 1)).fit(gate_rows()).get_feature_names_out()
    assert "last_content__word__<ext_email_1>" in names
    assert "proposed_call_args__word__<contact_1_email>" in names
    assert "history_text__word__<call>" in names and "history_text__word__<result>" in names
    assert not [n for n in names if n.endswith(("__ext_email_1", "__ext", "__email", "__contact_1_email"))]


def test_history_marker_with_a_space_is_one_token():
    X = gate_rows().assign(history_text="<history truncated>\n<call> a {}")
    names = TextBlock(view="gate", min_df=1, word_ngram=(1, 1)).fit(X).get_feature_names_out()
    assert "history_text__word__<history truncated>" in names
    assert "history_text__word__truncated" not in names


def test_char_ngrams_never_cut_a_placeholder():
    names = TextBlock(view="gate", analyzer="char", min_df=1).fit(gate_rows()).get_feature_names_out()
    grams = [n.split("__", 2)[2] for n in names]  # `<field>__char__<n-gram>`: check the n-gram only
    assert not [g for g in grams if "ext_" in g or "contact" in g or "<" in g or ">" in g]
    assert any(atomic_chars("<EXT_EMAIL_1>") in g for g in grams)


def test_placeholder_codes_are_collision_free_for_known_types_and_indices():
    codes = {atomic_chars(t) for t in KNOWN_TYPES}
    assert len(codes) == len(KNOWN_TYPES)
    indexed = {atomic_chars(f"<EXT_EMAIL_{k}>") for k in range(200)} | {atomic_chars(f"<CONTACT_{k}_EMAIL>")
                                                                       for k in range(200)}
    assert len(indexed) == 400
    assert all(PLACEHOLDER.fullmatch(t) for t in KNOWN_TYPES)


def test_placeholder_index_can_be_dropped():
    on = TextBlock(view="gate", min_df=1, word_ngram=(1, 1)).fit(gate_rows()).get_feature_names_out()
    off = TextBlock(view="gate", min_df=1, word_ngram=(1, 1), placeholder_index=False).fit(gate_rows())
    off = off.get_feature_names_out()
    assert {"last_content__word__<ext_email_1>", "last_content__word__<ext_email_2>"} <= set(on)
    assert "last_content__word__<ext_email>" in off and not [n for n in off if "<ext_email_" in n]
    assert Preprocessor("char", placeholder_index=False)("<EXT_EMAIL_3>") == atomic_chars("<EXT_EMAIL>")


def test_lowercase_never_touches_the_inside_of_a_char_code_and_can_be_disabled():
    assert Preprocessor("word")("URGENT <HEADER>") == "urgent <header>"
    assert Preprocessor("word", lowercase=False)("URGENT <HEADER>") == "URGENT <HEADER>"
    assert Preprocessor("char")("URGENT <HEADER>") == "urgent " + atomic_chars("<HEADER>")


def test_fields_are_vectorised_apart_and_weighted():
    X = gate_rows().assign(task="invoice", last_content="invoice")
    block = TextBlock(view="gate", min_df=1, word_ngram=(1, 1), fields=["task", "last_content"])
    names = list(block.fit(X).get_feature_names_out())
    assert "task__word__invoice" in names and "last_content__word__invoice" in names
    w = TextBlock(view="gate", min_df=1, word_ngram=(1, 1), fields=["task", "last_content"],
                  weights={"last_content": 0.5}).fit_transform(X).toarray()
    base = block.transform(X).toarray()
    i = names.index("last_content__word__invoice")
    np.testing.assert_allclose(w[:, i], 0.5 * base[:, i], atol=1e-6)
    assert block.vocabulary_sizes() == {"task__word": 1, "last_content__word": 1}


def test_fields_default_to_the_columns_given_by_the_harness():
    X = gate_rows()[["task", "proposed_call_tool", "proposed_call_args"]]  # gate "action only" variant
    block = TextBlock(view="gate", min_df=1).fit(X)
    assert block.fields_ == ["task", "proposed_call_tool", "proposed_call_args"]


def test_missing_text_reads_as_empty():
    X = gate_rows()
    X.loc[0, "last_content"] = None
    TextBlock(view="gate", min_df=1).fit_transform(X)


def test_svd_option_gives_dense_components():
    pipe = text_block("gate", svd=3, min_df=1)
    assert isinstance(pipe, Pipeline)
    out = pipe.fit_transform(gate_rows())
    assert out.shape == (6, 3) and isinstance(out, np.ndarray)


def test_cached_pipeline_fits_the_same_and_writes_the_cache(tmp_path):
    y = np.array([0, 1, 0, 1, 0, 1])
    a = tfidf_logreg("gate", text={"min_df": 1}).fit(gate_rows(), y).predict_proba(gate_rows())
    b = tfidf_logreg("gate", memory=str(tmp_path), text={"min_df": 1}).fit(gate_rows(), y).predict_proba(gate_rows())
    np.testing.assert_allclose(a, b)
    assert any(tmp_path.rglob("*"))


def test_usable_from_a_config_through_the_harness():
    from tests.test_experiment import toy
    df, folds = toy()
    for columns in (None, ["task", "proposed_call_tool", "proposed_call_args"]):
        c = E.ExperimentConfig(version="t", kind="model", views=["gate"], columns=columns,
                               model={"factory": "agentdrift.models.baselines:tfidf_logreg",
                                      "params": {"text": {"min_df": 1}}},
                               threshold_source="train", bootstrap={"n_resamples": 30, "seed": 42})
        data = E.make_dataset(df, "gate", c, folds)
        _, met = E.run_view(c, data)
        assert met["estimate"]["pr_auc"]["point"] > met["estimate"]["prevalence"]["point"] + 0.2


def test_a_field_without_any_token_gives_an_empty_block_not_an_error():
    X = gate_rows().assign(proposed_call_args="{}")
    block = TextBlock(view="gate", min_df=1)
    out = block.fit_transform(X)
    assert block.empty_fields_ == ["proposed_call_args__word"]
    assert block.vocabulary_sizes()["proposed_call_args__word"] == 0
    assert out.shape[1] == len(block.get_feature_names_out()) == block.transform(X).shape[1]
