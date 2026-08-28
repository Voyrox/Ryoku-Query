from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from dotenv import load_dotenv

from prowl import ProwlResult, SourceHit
from src.agent.answering import Answerer
from src.agent.ollama import OllamaClient
from support import SupportCard

ROUTES = frozenset({"gemma", "lfm"})
RISKS = frozenset({"informational", "state-changing", "destructive"})
_MAX_OUTPUT_CHARS = 1800
_THINKING_BLOCK = re.compile(r"<think\b.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINKING_MARKERS = re.compile(r"<think\b|</think\s*>|\bthinking\b|hidden-reasoning", re.IGNORECASE)


@dataclass(frozen=True)
class EvalEvidence:
    title: str
    answer: str
    risk: str
    docs_url: str | None


@dataclass(frozen=True)
class LLMEvalCase:
    id: str
    route: str
    question: str
    evidence: EvalEvidence
    sources: tuple[SourceHit, ...]
    required_answer_fragments: tuple[str, ...]
    forbidden_answer_fragments: tuple[str, ...]


@dataclass(frozen=True)
class LLMEvalResult:
    id: str
    route: str
    actual_route: str | None
    model: str | None
    passed: bool
    latency_ms: float
    answer: str
    missing_fragments: tuple[str, ...]
    forbidden_fragments: tuple[str, ...]
    reasoning_leak: bool


def _strict_keys(value: dict, allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _strings(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{context} must be a non-empty list of strings")
    return tuple(item.strip() for item in value)


def _source(row: object, context: str) -> SourceHit:
    if not isinstance(row, dict):
        raise TypeError(f"{context} must be an object")
    _strict_keys(row, {"file", "start_line", "end_line", "snippet", "risky"}, context)
    file = row.get("file")
    start = row.get("start_line")
    end = row.get("end_line")
    snippet = row.get("snippet")
    risky = row.get("risky", False)
    if not isinstance(file, str) or not file.strip() or Path(file).is_absolute() or ".." in Path(file).parts:
        raise ValueError(f"{context}.file must be a safe relative path")
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        raise ValueError(f"{context} must have valid start_line/end_line")
    if not isinstance(snippet, str) or not snippet.strip():
        raise ValueError(f"{context}.snippet must be a non-empty string")
    if not isinstance(risky, bool):
        raise ValueError(f"{context}.risky must be a boolean")
    return SourceHit(file.strip(), start, end, snippet.strip(), risky)


def load_llm_evals(path: Path) -> list[LLMEvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("LLM eval file must be a JSON object")
    _strict_keys(raw, {"schema_version", "cases"}, "LLM eval file")
    if raw.get("schema_version") != 1:
        raise ValueError("LLM eval schema_version must be 1")
    rows = raw.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("LLM eval cases must be a non-empty list")

    cases: list[LLMEvalCase] = []
    seen: set[str] = set()
    allowed = {
        "id",
        "route",
        "question",
        "evidence",
        "sources",
        "required_answer_fragments",
        "forbidden_answer_fragments",
    }
    evidence_allowed = {"title", "answer", "risk", "docs_url"}
    for index, row in enumerate(rows):
        context = f"case #{index}"
        if not isinstance(row, dict):
            raise TypeError(f"{context} must be an object")
        _strict_keys(row, allowed, context)
        case_id = row.get("id")
        route = row.get("route")
        question = row.get("question")
        evidence = row.get("evidence")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{context}.id must be a non-empty string")
        if case_id in seen:
            raise ValueError(f"duplicate LLM eval id: {case_id}")
        seen.add(case_id)
        if route not in ROUTES:
            raise ValueError(f"case {case_id} has invalid route: {route}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"case {case_id}.question must be a non-empty string")
        if not isinstance(evidence, dict):
            raise TypeError(f"case {case_id}.evidence must be an object")
        _strict_keys(evidence, evidence_allowed, f"case {case_id}.evidence")
        title = evidence.get("title")
        answer = evidence.get("answer")
        risk = evidence.get("risk")
        docs_url = evidence.get("docs_url")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"case {case_id}.evidence.title must be non-empty")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"case {case_id}.evidence.answer must be non-empty")
        if risk not in RISKS:
            raise ValueError(f"case {case_id}.evidence.risk is invalid")
        if docs_url is not None and (
            not isinstance(docs_url, str) or not docs_url.startswith("https://")
        ):
            raise ValueError(f"case {case_id}.evidence.docs_url must be HTTPS or null")
        sources = row.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError(f"case {case_id}.sources must be a list")
        if route == "lfm" and not sources:
            raise ValueError(f"case {case_id} route lfm requires cited sources")
        if route == "gemma" and sources:
            raise ValueError(f"case {case_id} route gemma must not include sources")
        cases.append(
            LLMEvalCase(
                id=case_id.strip(),
                route=route,
                question=question.strip(),
                evidence=EvalEvidence(title.strip(), answer.strip(), risk, docs_url),
                sources=tuple(_source(source, f"case {case_id}.sources[{i}]") for i, source in enumerate(sources)),
                required_answer_fragments=_strings(
                    row.get("required_answer_fragments"),
                    f"case {case_id}.required_answer_fragments",
                ),
                forbidden_answer_fragments=tuple(
                    item.strip()
                    for item in row.get("forbidden_answer_fragments", [])
                    if isinstance(item, str) and item.strip()
                ),
            )
        )
    return cases


def leaks_hidden_reasoning(text: str) -> bool:
    return bool(_THINKING_MARKERS.search(text))


def safe_visible_answer(text: str) -> str:
    safe = _THINKING_BLOCK.sub("[redacted]", text)
    safe = _THINKING_MARKERS.sub("[redacted]", safe)
    return safe.strip()[:_MAX_OUTPUT_CHARS]


def card_for_case(case: LLMEvalCase) -> SupportCard:
    return SupportCard(
        id=f"llm-eval.{case.id}",
        title=case.evidence.title,
        examples=(case.question, f"eval {case.id}"),
        keywords=(),
        exact_terms=(),
        answer=case.evidence.answer,
        risk=case.evidence.risk,
        docs_url=case.evidence.docs_url,
        source_hints=tuple(hit.file for hit in case.sources),
        clarifies_with=(),
    )


def sources_for_case(case: LLMEvalCase) -> ProwlResult | None:
    if not case.sources:
        return None
    return ProwlResult("ok", hits=case.sources)


def evaluate_text(
    case: LLMEvalCase,
    text: str,
    *,
    actual_route: str | None,
    model: str | None,
    latency_ms: float,
) -> LLMEvalResult:
    lowered = text.lower()
    missing = tuple(
        fragment
        for fragment in case.required_answer_fragments
        if fragment.lower() not in lowered
    )
    forbidden = tuple(
        fragment
        for fragment in case.forbidden_answer_fragments
        if fragment.lower() in lowered
    )
    leaked = leaks_hidden_reasoning(text)
    passed = (
        actual_route == case.route
        and not missing
        and not forbidden
        and not leaked
    )
    return LLMEvalResult(
        id=case.id,
        route=case.route,
        actual_route=actual_route,
        model=model,
        passed=passed,
        latency_ms=latency_ms,
        answer=safe_visible_answer(text),
        missing_fragments=tuple(safe_visible_answer(fragment) for fragment in missing),
        forbidden_fragments=tuple(safe_visible_answer(fragment) for fragment in forbidden),
        reasoning_leak=leaked,
    )


async def run_case(
    case: LLMEvalCase,
    answerer: Answerer,
    *,
    gemma_model: str,
    lfm_model: str,
) -> LLMEvalResult:
    started = time.perf_counter()
    rendered = await answerer.render(case.question, card_for_case(case), sources_for_case(case))
    latency_ms = (time.perf_counter() - started) * 1000
    model = None
    if rendered.route == "gemma":
        model = gemma_model
    elif rendered.route == "lfm":
        model = lfm_model
    return evaluate_text(
        case,
        rendered.text,
        actual_route=rendered.route,
        model=model,
        latency_ms=latency_ms,
    )


async def run_benchmark(
    cases: Sequence[LLMEvalCase],
    answerer: Answerer,
    *,
    gemma_model: str,
    lfm_model: str,
) -> list[LLMEvalResult]:
    results: list[LLMEvalResult] = []
    for case in cases:
        results.append(
            await run_case(
                case,
                answerer,
                gemma_model=gemma_model,
                lfm_model=lfm_model,
            )
        )
    return results


def report_payload(results: Sequence[LLMEvalResult]) -> dict:
    passed = sum(result.passed for result in results)
    latencies = [result.latency_ms for result in results]
    return {
        "case_count": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "mean_ms": fmean(latencies) if latencies else 0.0,
        "results": [asdict(result) for result in results],
    }


def print_human(results: Sequence[LLMEvalResult]) -> None:
    payload = report_payload(results)
    print(
        f"llm: {payload['passed']}/{payload['case_count']} cases "
        f"pass_rate={payload['pass_rate']:.1%} mean={payload['mean_ms']:.2f}ms"
    )
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        route = result.actual_route or "fallback"
        print(f"  {status} {result.id} route={route} model={result.model or '-'} {result.latency_ms:.2f}ms")
        if not result.passed:
            if result.missing_fragments:
                print(f"    missing: {', '.join(result.missing_fragments)}")
            if result.forbidden_fragments:
                print(f"    forbidden: {', '.join(result.forbidden_fragments)}")
            if result.reasoning_leak:
                print("    leaked hidden reasoning marker")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", type=Path, default=Path("data/llm-evals.json"))
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--gemma-model", default=None)
    parser.add_argument("--lfm-model", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    host = (args.ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    gemma_model = args.gemma_model or os.getenv("GEMMA_MODEL", "gemma4:e4b")
    lfm_model = args.lfm_model or os.getenv("LFM_MODEL", "lfm2.5:latest")
    timeout = args.timeout if args.timeout is not None else float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    cases = load_llm_evals(args.evals)
    answerer = Answerer(
        OllamaClient(
            host,
            gemma_model,
            lfm_model,
            timeout=timeout,
        )
    )
    results = asyncio.run(
        run_benchmark(
            cases,
            answerer,
            gemma_model=gemma_model,
            lfm_model=lfm_model,
        )
    )
    if args.json:
        print(json.dumps(report_payload(results), indent=2, sort_keys=True))
    else:
        print_human(results)
    return 1 if args.check and not all(result.passed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
