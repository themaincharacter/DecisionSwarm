"""Model loaders: how a cell's worker builds its predictor.

A loader adapts one model family to the worker's predict interface:

``predict(views, questions) -> (answers, warnings)``

where ``views`` is the cell's bound views (name -> payload), ``questions`` are
resolved :class:`~decision_swarm.manifest.QuestionSpec` objects, and answers
are wire dicts (``{question: {"<type>": value, "confidence": ...}}``).

- ``load``: PyPI ``laya`` — ``laya.load(repo, subfolder=...)`` over text views.
- ``load_vlm``: the ``laya-vision`` fork — ``laya.load_vlm(repo)`` with an
  image view (opened with PIL) plus optional text views as context.
- ``stub``: deterministic offline predictor for tests and demos; echoes the
  view names it received in a warning so view isolation is observable.

Each loader runs inside the env built for its cell's ``package_source``. Both
real loaders prefer a checkpoint already present in the shared model cache over
the Hub repo id, so ``fetch`` (and ``--offline``) actually control what is loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from decision_swarm.manifest import QuestionSpec, question_to_dict


class LoaderError(RuntimeError):
    """Raised when a loader cannot be resolved or a prediction cannot be built."""


@dataclass(frozen=True)
class ModelSpec:
    """Where a cell's checkpoint lives and how to address it."""

    model_id: str
    hf_repo: str | None = None
    subfolder: str | None = None
    revision: str = "main"
    models_dir: str | None = None

    @property
    def repo_id(self) -> str:
        return self.hf_repo or self.model_id

    @property
    def local_path(self) -> Path | None:
        """The cached snapshot directory, when the checkpoint is really there.

        Mirrors the cache layout (``<models-dir>/<repo>/<revision>/`` plus the
        optional subfolder) and requires a ``config.json`` so a half-downloaded
        directory falls back to the Hub instead of failing to load.
        """
        if not (self.models_dir and self.hf_repo and self.revision):
            return None
        snapshot = Path(self.models_dir) / self.hf_repo / self.revision
        target = snapshot / self.subfolder if self.subfolder else snapshot
        return target if (target / "config.json").is_file() else None



class Predictor(Protocol):
    """Answers one cell's questions over its bound views."""

    def predict(
        self, views: Mapping[str, str], questions: Mapping[str, QuestionSpec]
    ) -> tuple[Mapping[str, Any], tuple[str, ...]]: ...


class StubPredictor:
    """Deterministic offline predictor: one typed answer per question."""

    def __init__(self, view_types: Mapping[str, str] | None = None) -> None:
        self._view_types = dict(view_types or {})

    def predict(
        self, views: Mapping[str, str], questions: Mapping[str, QuestionSpec]
    ) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        answers: dict[str, Any] = {}
        for name, question in questions.items():
            if question.type == "choice":
                criteria = question.criteria or {}
                options = list(criteria) if isinstance(criteria, dict) else list(criteria)
                answers[name] = {"choice": options[0] if options else None, "confidence": 1.0}
            elif question.type == "score":
                answers[name] = {"score": 0.0, "confidence": 1.0}
            else:
                answers[name] = {"noul": 0.5, "confidence": 1.0}
        warnings = (f"views={','.join(sorted(views))}",)
        return answers, warnings


class _TextPredictor:
    """PyPI ``laya``: a dict state built from the cell's bound views."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def predict(
        self, views: Mapping[str, str], questions: Mapping[str, QuestionSpec]
    ) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        state: Any = dict(views) if len(views) != 1 else next(iter(views.values()))
        result = self._agent.predict(state, {name: question_to_dict(q) for name, q in questions.items()})
        return result["answers"], ()


class _VisionPredictor:
    """``laya-vision``: one image view as the picture, text views as context."""

    def __init__(self, agent: Any, view_types: Mapping[str, str]) -> None:
        self._agent = agent
        self._view_types = dict(view_types)

    def predict(
        self, views: Mapping[str, str], questions: Mapping[str, QuestionSpec]
    ) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        from PIL import Image

        image_views = [name for name in views if self._view_types.get(name) == "image"]
        if not image_views:
            raise LoaderError("no image view is bound to this cell")
        state: dict[str, Any] = {"image": Image.open(views[image_views[0]])}
        notes = [value for name, value in views.items() if self._view_types.get(name) == "text"]
        if notes:
            state["note"] = "\n".join(notes)
        warnings: tuple[str, ...] = ()
        if len(image_views) > 1:
            warnings = (f"multiple image views bound; using {image_views[0]!r}",)
        result = self._agent.predict(state, {name: question_to_dict(q) for name, q in questions.items()})
        return result["answers"], warnings


def load_laya(spec: ModelSpec, view_types: Mapping[str, str]) -> Predictor:
    """PyPI ``laya`` single-model loader (``laya.load``)."""
    import laya  # provided by this cell's env (package_source)

    local = spec.local_path
    if local is not None:
        # the subfolder is already part of the resolved path
        return _TextPredictor(laya.load(str(local)))
    kwargs = {"subfolder": spec.subfolder} if spec.subfolder else {}
    return _TextPredictor(laya.load(spec.repo_id, **kwargs))


def load_laya_vlm(spec: ModelSpec, view_types: Mapping[str, str]) -> Predictor:
    """``laya-vision`` fork loader (``laya.load_vlm``)."""
    import laya  # the fork's package, in this cell's env

    local = spec.local_path
    if local is not None:
        return _VisionPredictor(laya.load_vlm(str(local)), view_types)
    return _VisionPredictor(laya.load_vlm(spec.repo_id, revision=spec.revision), view_types)


def load_stub(spec: ModelSpec, view_types: Mapping[str, str]) -> Predictor:
    """Deterministic offline loader for tests and demos."""
    return StubPredictor(view_types)


LOADERS: dict[str, Callable[[ModelSpec, Mapping[str, str]], Predictor]] = {
    "load": load_laya,
    "load_vlm": load_laya_vlm,
    "stub": load_stub,
}
"""Loader registry — keys match the model registry's ``loader`` field."""


def build_predictor(
    loader: str, spec: ModelSpec, view_types: Mapping[str, str] | None = None
) -> Predictor:
    """Build the predictor for a loader name (``load``, ``load_vlm``, ``stub``)."""
    try:
        factory = LOADERS[loader]
    except KeyError:
        raise LoaderError(f"unknown loader {loader!r} (known: {', '.join(sorted(LOADERS))})") from None
    return factory(spec, view_types or {})
