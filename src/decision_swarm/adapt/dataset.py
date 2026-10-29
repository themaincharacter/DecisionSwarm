"""Loading and validation of adaptation datasets.

Rows follow the schema ``laya.train`` consumes: ``{state, questions, gold}`` or
``{state, questions, expected}`` — one JSON object per JSONL line, or one CSV
row per labelled question.  Validation here is pure Python (no model stack) and
mirrors how ``laya.train.items_from_rows`` will read the same rows: the skip
reasons match its vocabulary, target parseability follows
``target_from_gold`` / ``target_from_expected``, and train/eval leakage is
detected by (state, question) signatures the way ``laya.train.finetune`` does.

It deliberately does *not* validate question schemas against a pack: the
adapter's ``questions.json`` is optional default question material, never
enforced.  See ``docs/agents/adapt.md`` for dataset recipes.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from decision_swarm.manifest import QUESTION_TYPES

COLLAPSE_SHARE = 0.9
"""A question whose top label holds at least this share of its items is flagged.

A model can score high on a collapsed question by ignoring the state entirely;
the fix is more (and more negative) examples — see ``docs/agents/adapt.md``.
"""

ROW_SKIP_REASONS = ("invalid_json", "invalid_row", "empty_text", "invalid_questions")
"""Skip reasons counted once per row (``empty_text`` matches ``items_from_rows``)."""

ITEM_SKIP_REASONS = ("invalid_question", "empty_label", "invalid_target")
"""Skip reasons counted per labelled question (same names as ``items_from_rows``)."""


class DatasetError(ValueError):
    """Raised when a dataset file cannot be read at all."""


# ----------------------------------------------------------------- canonical forms


def _canonical(value: Any) -> str:
    """Stable text for signatures: the same facts hash the same way."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _is_seq(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


class _BadRow:
    """A row that never became a record, carrying its skip reason."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason


# ----------------------------------------------------------------- question parsing


@dataclass(frozen=True)
class _Options:
    """How targets index a question, mirroring ``target_from_gold``/``expected``."""

    gold_keys: tuple[Any, ...]
    expected_keys: tuple[Any, ...]


def _parse_question(raw: Any) -> tuple[_Options | None, str | None]:
    """``(options, None)`` for a trainable question, else ``(None, reason)``.

    Mirrors the tolerances ``laya.train.to_internal`` applies: a question is an
    object with ``type`` and non-blank ``instructions``; choice and score need
    at least two criteria, noul takes none.
    """
    if not isinstance(raw, Mapping):
        return None, "invalid_question"
    qtype = raw.get("type")
    if qtype not in QUESTION_TYPES:
        return None, "invalid_question"
    instructions = raw.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        return None, "invalid_question"
    criteria = raw.get("criteria")
    if qtype == "noul":
        if criteria is not None:
            return None, "invalid_question"
        return _Options(("false", "true"), ("false", "true")), None
    if isinstance(criteria, Mapping):
        keys = tuple(criteria.keys())
    elif _is_seq(criteria):
        keys = tuple(criteria)
    else:
        return None, "invalid_question"
    if len(keys) < 2 or any(not isinstance(k, (str, int, float)) or isinstance(k, bool) for k in keys):
        return None, "invalid_question"
    if qtype == "choice":
        # gold reads the criterion keys; expected matches them raw or as text.
        return _Options(keys, keys), None
    # score: gold is always keyed by level index text, even for dict criteria;
    # expected matches dict keys (or index text for list criteria) or a level.
    expected_keys = keys if isinstance(criteria, Mapping) else tuple(str(i) for i in range(len(keys)))
    return _Options(tuple(str(i) for i in range(len(keys))), expected_keys), None


# ----------------------------------------------------------------- target parsing


def _probability_map(raw: Any, options: _Options) -> tuple[dict[Any, float] | None, set[str], str | None]:
    """``gold`` parse: the ``probabilities`` map, dropped keys, and a reason.

    Mirrors ``target_from_gold``: missing keys are zero (noul defaults to 0.5),
    unknown keys are silently dropped by the trainer, and negative or
    non-finite values are invalid.
    """
    if not isinstance(raw, Mapping) or not isinstance(raw.get("probabilities"), Mapping):
        return None, set(), "invalid_target"
    probs = raw["probabilities"]
    values: dict[Any, float] = {}
    for key in options.gold_keys:
        got = probs.get(key, probs.get(str(key)))
        if got is None:
            values[key] = 0.5 if key in ("false", "true") else 0.0
            continue
        try:
            number = float(got)
        except (TypeError, ValueError):
            return None, set(), "invalid_target"
        if not (-float("inf") < number < float("inf")) or number < 0:
            return None, set(), "invalid_target"
        values[key] = number
    matched = {key for key in probs if key in options.gold_keys or key in {str(k) for k in options.gold_keys}}
    dropped = {str(key) for key in probs} - {str(key) for key in matched}
    return values, dropped, None


def _expected_index(raw: Any, options: _Options, qtype: str) -> tuple[str | None, str | None]:
    """``expected`` parse: the normalized label it resolves to, and a reason.

    Mirrors ``target_from_expected``: a choice label (raw key or its text), a
    noul boolean (``bool``, 0/1, or true/false-ish text), a score level (key,
    index text, or a number in ``[0, K-1]``).
    """
    if isinstance(raw, str) and not raw.strip():
        return None, "empty_label"
    if qtype == "noul":
        if isinstance(raw, bool):
            return ("true" if raw else "false"), None
        if isinstance(raw, (int, float)):
            if raw in (0, 1):
                return ("true" if raw else "false"), None
            return None, "invalid_target"
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text in ("true", "1", "yes"):
                return "true", None
            if text in ("false", "0", "no"):
                return "false", None
        return None, "invalid_target"
    if qtype == "choice":
        for key in options.expected_keys:
            if raw == key or str(raw) == str(key):
                return str(key), None
        return None, "invalid_target"
    # score: a level key (dict key or index text), or a number in [0, K-1].
    for key in options.expected_keys:
        if raw == key or str(raw) == str(key):
            return str(key), None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "invalid_target"
    if isinstance(raw, bool) or not (0.0 <= value <= float(len(options.gold_keys) - 1)):
        return None, "invalid_target"
    if value == int(value):
        return str(int(value)), None
    return str(value), None


def _gold_label(values: Mapping[Any, float]) -> str:
    """The label a gold distribution points at: its argmax, or ``(uniform)``."""
    total = sum(values.values())
    if total <= 0.0:
        return "(uniform)"
    best = max(values.values())
    for key, value in values.items():
        if value == best:
            return str(key)
    return "(uniform)"  # pragma: no cover — max() always finds its value


# ----------------------------------------------------------------- reports


@dataclass
class QuestionStats:
    """Label balance for one question over its validated items."""

    qid: str
    type: str
    declared: int = 0
    items: int = 0
    labels: Counter = field(default_factory=Counter)
    dropped_keys: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "type": self.type,
            "declared": self.declared,
            "items": self.items,
            "labels": dict(self.labels),
            "dropped_keys": self.dropped_keys,
        }


@dataclass
class DatasetReport:
    """What ``adapt validate`` knows about one dataset file."""

    source: str
    rows: int = 0
    items: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    questions: dict[str, QuestionStats] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    signature_counts: Counter = field(default_factory=Counter)

    @property
    def duplicates(self) -> int:
        """Items repeating an earlier (state, question) signature in this file."""
        return self.items - len(self.signature_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows": self.rows,
            "items": self.items,
            "skipped": dict(self.skipped),
            "questions": [stats.to_dict() for stats in self.questions.values()],
            "warnings": list(self.warnings),
            "duplicates": self.duplicates,
        }


def leaked_items(eval_report: DatasetReport, train_report: DatasetReport) -> int:
    """Eval items whose (state, question) signature also occurs in train.

    Same inputs with (possibly) different labels — the leak ``laya.train`` warns
    about, measured here before any training run.
    """
    train_sigs = train_report.signature_counts
    return sum(count for sig, count in eval_report.signature_counts.items() if sig in train_sigs)


# ----------------------------------------------------------------- loading


def rows_from_csv(
    path: str | Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
    question_id: str = "label",
    instructions: str | None = None,
) -> list[dict[str, Any]]:
    """One CSV row per labelled question, as ``laya.train.rows_from_csv`` builds it.

    Distinct labels in ``label_column`` (order of appearance) become the options
    of one choice question.  ``utf-8-sig`` so Excel's BOM is handled.  A state
    richer than one text column needs JSONL.
    """
    path = Path(path)
    rows_raw: list[dict[str, str]] = []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise DatasetError(f"{path} is an empty CSV file")
            if text_column not in reader.fieldnames:
                raise DatasetError(
                    f"CSV file {path} missing text column {text_column!r} (available: {reader.fieldnames!r})"
                )
            if label_column not in reader.fieldnames:
                raise DatasetError(
                    f"CSV file {path} missing label column {label_column!r} (available: {reader.fieldnames!r})"
                )
            rows_raw = list(reader)
    except OSError as exc:
        raise DatasetError(f"{path}: {exc}") from exc
    if not rows_raw:
        raise DatasetError(f"{path} has no rows")

    labels: list[str] = []
    seen: set[str] = set()
    for row in rows_raw:
        value = (row.get(label_column) or "").strip()
        if value and value not in seen:
            seen.add(value)
            labels.append(value)
    if len(labels) < 2:
        raise DatasetError(
            f"CSV file {path} must contain at least 2 distinct labels in {label_column!r}, found {labels!r}"
        )

    questions = {
        question_id: {
            "type": "choice",
            "instructions": instructions or "Classify the input into the correct category.",
            "criteria": labels,
        }
    }
    return [
        {
            "state": row.get(text_column, ""),
            "questions": questions,
            "expected": {question_id: (row.get(label_column) or "").strip()},
        }
        for row in rows_raw
    ]


def read_jsonl(path: str | Path) -> list[Any]:
    """All records in a JSONL file; malformed lines become ``_BadRow`` marks.

    Blank lines and ``#`` comments are ignored, like ``laya.train.read_jsonl``.
    """
    path = Path(path)
    records: list[Any] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    records.append(_BadRow("invalid_json"))
                else:
                    records.append(record)
    except OSError as exc:
        raise DatasetError(f"{path}: {exc}") from exc
    return records


def read_rows(
    path: str | Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
    question_id: str = "label",
    instructions: str | None = None,
) -> list[Any]:
    """Training/eval rows from a JSONL or CSV file (by extension)."""
    if str(path).lower().endswith(".csv"):
        return rows_from_csv(
            path,
            text_column=text_column,
            label_column=label_column,
            question_id=question_id,
            instructions=instructions,
        )
    return read_jsonl(path)


# ----------------------------------------------------------------- validation


def _skip(report: DatasetReport, reason: str) -> None:
    report.skipped[reason] = report.skipped.get(reason, 0) + 1


def validate_rows(records: Iterable[Any], *, source: str = "<rows>") -> DatasetReport:
    """Validate rows into a :class:`DatasetReport` (items, skips, balance).

    Unreadable rows are skipped with a reason, never fatal — this is a dry run.
    Questions with neither ``gold`` nor ``expected`` produce no items and no
    skips, exactly as ``items_from_rows`` treats them.
    """
    report = DatasetReport(source=source)
    for record in records:
        if isinstance(record, _BadRow):
            report.rows += 1
            _skip(report, record.reason)
            continue
        if not isinstance(record, Mapping):
            report.rows += 1
            _skip(report, "invalid_row")
            continue
        report.rows += 1
        state = record.get("state")
        if state is None or (isinstance(state, str) and not state.strip()):
            _skip(report, "empty_text")
            continue
        questions = record.get("questions", {})
        if not isinstance(questions, Mapping):
            _skip(report, "invalid_questions")
            continue
        gold = record.get("gold") if isinstance(record.get("gold"), Mapping) else None
        expected = record.get("expected") if isinstance(record.get("expected"), Mapping) else None
        state_text = _canonical(state)
        for qid, raw_question in questions.items():
            qid = str(qid)
            stats = report.questions.setdefault(qid, QuestionStats(qid=qid, type="?"))
            stats.declared += 1
            if isinstance(raw_question, Mapping) and isinstance(raw_question.get("type"), str):
                stats.type = raw_question["type"]
            gold_target = gold.get(qid) if gold is not None else None
            expected_target = expected.get(qid) if expected is not None else None
            if gold_target is None and expected_target is None:
                continue  # unlabelled: no item, no skip (items_from_rows semantics)
            options, reason = _parse_question(raw_question)
            if reason is not None or options is None:
                _skip(report, "invalid_question")
                continue
            qtype = raw_question["type"]
            label: str | None = None
            if gold_target is not None:
                values, dropped, reason = _probability_map(gold_target, options)
                if reason is not None or values is None:
                    _skip(report, "invalid_target")
                    continue
                if dropped:
                    stats.dropped_keys += len(dropped)
                    report.warnings.append(
                        f"question {qid!r}: gold probabilities name {len(dropped)} unknown "
                        f"option(s) {sorted(dropped)} — their mass is dropped by the trainer"
                    )
                label = _gold_label(values)
            else:
                label, reason = _expected_index(expected_target, options, qtype)
                if reason is not None or label is None:
                    _skip(report, reason or "invalid_target")
                    continue
            stats.items += 1
            stats.labels[label] += 1
            report.items += 1
            report.signature_counts[sha256(
                _canonical([state_text, qid, _canonical(raw_question)]).encode("utf-8")
            ).hexdigest()] += 1

    for stats in report.questions.values():
        if stats.items == 0:
            report.warnings.append(
                f"question {stats.qid!r} is declared in {stats.declared} row(s) but never labelled"
            )
        elif stats.labels:
            top_label, top_count = stats.labels.most_common(1)[0]
            share = top_count / stats.items
            if share >= COLLAPSE_SHARE:
                report.warnings.append(
                    f"question {stats.qid!r} is {share:.0%} label {top_label!r} "
                    f"({top_count}/{stats.items} items) — a model can score high without reading "
                    f"the state; add negatives (docs/agents/adapt.md)"
                )
    if report.duplicates > 0:
        report.warnings.append(
            f"{report.duplicates} item(s) repeat an earlier (state, question) signature — "
            f"duplicates inflate one example's weight"
        )
    return report
