"""Tests for adaptation dataset loading and validation (pure Python, no laya)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_swarm.adapt import budget, dataset

CHOICE = {"type": "choice", "instructions": "Which department?", "criteria": {"billing": "Billing", "tech": "Tech"}}
NOUL = {"type": "noul", "instructions": "Is it urgent?"}
SCORE = {"type": "score", "instructions": "How severe?", "criteria": ["low", "high"]}


def _row(state="ticket text", questions=None, **targets):
    row = {"state": state, "questions": questions if questions is not None else {"q": CHOICE}}
    row.update(targets)
    return row


def test_gold_rows_produce_items_with_argmax_labels() -> None:
    report = dataset.validate_rows(
        [
            _row(gold={"q": {"probabilities": {"billing": 0.9, "tech": 0.1}}}),
            _row(state="other", gold={"q": {"probabilities": {"billing": 0.2, "tech": 0.8}}}),
        ]
    )
    assert report.rows == 2
    assert report.items == 2
    assert report.skipped == {}
    assert report.questions["q"].labels == {"billing": 1, "tech": 1}


def test_expected_targets_resolve_labels_per_type() -> None:
    # score expected labels follow laya.train: index text for list criteria,
    # criterion keys for dict criteria (Agent._to_internal leaves score as-is).
    score_dict = {"type": "score", "instructions": "How severe?", "criteria": {"low": "Low", "high": "High"}}
    report = dataset.validate_rows(
        [
            _row(questions={"d": CHOICE}, expected={"d": "tech"}),
            _row(questions={"u": NOUL}, expected={"u": True}),
            _row(questions={"u": NOUL}, expected={"u": "no"}),
            _row(questions={"s": SCORE}, expected={"s": 1}),
            _row(questions={"s": SCORE}, expected={"s": "0"}),
            _row(questions={"sd": score_dict}, expected={"sd": "high"}),
        ]
    )
    assert report.items == 6
    assert report.questions["d"].labels == {"tech": 1}
    assert report.questions["u"].labels == {"true": 1, "false": 1}
    assert report.questions["s"].labels == {"1": 1, "0": 1}
    assert report.questions["sd"].labels == {"high": 1}


def test_skip_reasons_mirror_items_from_rows() -> None:
    report = dataset.validate_rows(
        [
            {"state": "  ", "questions": {"q": CHOICE}, "expected": {"q": "billing"}},  # empty_text
            _row(questions={"q": {"type": "choice", "instructions": "x", "criteria": []}},
                 expected={"q": "billing"}),  # invalid_question
            _row(questions={"q": CHOICE}, expected={"q": "nope"}),  # invalid_target
            _row(questions={"q": CHOICE}, expected={"q": "   "}),  # empty_label
            _row(questions={"q": CHOICE}),  # unlabelled: no item, no skip
        ]
    )
    assert report.rows == 5
    assert report.items == 0
    assert report.skipped == {
        "empty_text": 1,
        "invalid_question": 1,
        "invalid_target": 1,
        "empty_label": 1,
    }


def test_row_level_skip_reasons() -> None:
    records = dataset.read_jsonl  # exercised through a file below
    assert callable(records)
    report = dataset.validate_rows(
        [dataset._BadRow("invalid_json"), "not-an-object",
         {"questions": {"q": CHOICE}, "expected": {"q": "billing"}}]
    )
    assert report.rows == 3
    assert report.skipped == {"invalid_json": 1, "invalid_row": 1, "empty_text": 1}


def test_gold_probabilities_parse_like_target_from_gold() -> None:
    uniform = dataset.validate_rows([_row(gold={"q": {"probabilities": {}}})])
    assert uniform.items == 1
    assert uniform.questions["q"].labels == {"(uniform)": 1}

    negative = dataset.validate_rows(
        [_row(gold={"q": {"probabilities": {"billing": -2.0}}})]
    )
    assert negative.items == 0
    assert negative.skipped == {"invalid_target": 1}

    missing = dataset.validate_rows([_row(gold={"q": {}})])
    assert missing.skipped == {"invalid_target": 1}


def test_gold_unknown_keys_warn_about_dropped_mass() -> None:
    report = dataset.validate_rows(
        [_row(gold={"q": {"probabilities": {"billing": 0.5, "billng": 0.5}}})]
    )
    assert report.questions["q"].dropped_keys == 1
    assert any("unknown" in w and "billng" in w for w in report.warnings)


def test_collapse_warning_fires_on_a_100_row_subset() -> None:
    rows = [_row(questions={"u": NOUL}, expected={"u": i < 95}) for i in range(100)]
    report = dataset.validate_rows(rows)
    assert report.items == 100
    assert any("95%" in w and "u" in w for w in report.warnings)


def test_never_labelled_question_warns() -> None:
    report = dataset.validate_rows([_row(questions={"q": CHOICE, "ghost": NOUL}, expected={"q": "billing"})])
    assert report.items == 1
    assert any("ghost" in w and "never labelled" in w for w in report.warnings)


def test_leakage_and_duplicates_use_state_question_signatures() -> None:
    shared = _row(state="same ticket", questions={"q": CHOICE}, expected={"q": "billing"})
    train = dataset.validate_rows([shared, shared, _row(state="other", expected={"q": "tech"})])
    assert train.items == 3
    assert train.duplicates == 1
    eval_report = dataset.validate_rows(
        [shared, _row(state="fresh", expected={"q": "tech"})]
    )
    assert dataset.leaked_items(eval_report, train) == 1


def test_jsonl_file_loading_skips_bad_lines_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "# a comment\n\n"
        + json.dumps(_row(expected={"q": "billing"})) + "\n"
        + "{not json\n"
        + json.dumps(_row(expected={"q": "tech"})) + "\n",
        encoding="utf-8",
    )
    records = dataset.read_jsonl(path)
    report = dataset.validate_rows(records, source=str(path))
    assert report.rows == 3  # the malformed line counts as a row
    assert report.items == 2
    assert report.skipped == {"invalid_json": 1}


def test_csv_becomes_one_choice_question(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("\ufefftext,label\nhello,billing\nworld,tech\nhello,billing\n", encoding="utf-8")
    rows = dataset.rows_from_csv(path)
    assert len(rows) == 3
    assert rows[0]["questions"]["label"]["criteria"] == ["billing", "tech"]
    report = dataset.validate_rows(rows, source=str(path))
    assert report.items == 3
    assert report.questions["label"].labels == {"billing": 2, "tech": 1}


def test_csv_errors_match_upstream(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="empty CSV"):
        dataset.rows_from_csv(empty)

    missing = tmp_path / "missing.csv"
    missing.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="missing text column"):
        dataset.rows_from_csv(missing)

    one_label = tmp_path / "one.csv"
    one_label.write_text("text,label\nhello,billing\nworld,billing\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="at least 2 distinct labels"):
        dataset.rows_from_csv(one_label)


def test_read_rows_dispatches_csv_by_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("text,label\nhello,billing\nworld,tech\n", encoding="utf-8")
    rows = dataset.read_rows(path)
    assert len(rows) == 2
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(json.dumps(_row(expected={"q": "billing"})) + "\n", encoding="utf-8")
    assert len(dataset.read_rows(jsonl)) == 1


def test_budget_formulas_mirror_laya_train() -> None:
    assert budget.split_counts(500) == (450, 50)
    assert budget.split_counts(10_000) == (9_600, 400)  # calib capped at 400
    assert budget.optimizer_updates(450) == 32  # ceil(450/8)=57 -> ceil(57/8)=8 -> x4 epochs
    message = budget.budget_message(450)
    assert message.startswith("optimizer budget: 450 train item(s)")
    assert "32 optimizer updates over 4 epoch(s)" in message


def test_report_to_dict_is_json_safe(tmp_path: Path) -> None:
    report = dataset.validate_rows([_row(expected={"q": "billing"})], source="train.jsonl")
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["source"] == "train.jsonl"
    assert payload["items"] == 1
    assert payload["questions"][0]["qid"] == "q"
