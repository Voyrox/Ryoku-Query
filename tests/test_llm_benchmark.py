import json
import tempfile
import unittest
from pathlib import Path

from benchmark_llm import (
    EvalEvidence,
    LLMEvalCase,
    evaluate_text,
    leaks_hidden_reasoning,
    load_llm_evals,
    report_payload,
    safe_visible_answer,
)
from prowl import SourceHit


class LLMBenchmarkSchemaTests(unittest.TestCase):
    def write_eval(self, document):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "llm-evals.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def valid_case(self, **overrides):
        case = {
            "id": "gemma-doctor",
            "route": "gemma",
            "question": "How do I run doctor?",
            "evidence": {
                "title": "Run Ryoku Doctor",
                "answer": "Run `ryoku doctor`.",
                "risk": "informational",
                "docs_url": "https://docs.ryoku.dev/docs/troubleshoot",
            },
            "sources": [],
            "required_answer_fragments": ["ryoku doctor"],
            "forbidden_answer_fragments": ["<think>", "thinking"],
        }
        case.update(overrides)
        return case

    def test_loads_valid_gemma_case(self):
        path = self.write_eval({"schema_version": 1, "cases": [self.valid_case()]})

        cases = load_llm_evals(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].id, "gemma-doctor")
        self.assertEqual(cases[0].route, "gemma")
        self.assertEqual(cases[0].required_answer_fragments, ("ryoku doctor",))

    def test_lfm_case_requires_cited_sources(self):
        path = self.write_eval(
            {
                "schema_version": 1,
                "cases": [self.valid_case(id="lfm", route="lfm")],
            }
        )

        with self.assertRaisesRegex(ValueError, "requires cited sources"):
            load_llm_evals(path)

    def test_gemma_case_rejects_sources(self):
        path = self.write_eval(
            {
                "schema_version": 1,
                "cases": [
                    self.valid_case(
                        sources=[
                            {
                                "file": "ryoku/cli/main.go",
                                "start_line": 1,
                                "end_line": 20,
                                "snippet": "package main",
                            }
                        ]
                    )
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "must not include sources"):
            load_llm_evals(path)

    def test_rejects_missing_required_fragments(self):
        case = self.valid_case()
        case.pop("required_answer_fragments")
        path = self.write_eval({"schema_version": 1, "cases": [case]})

        with self.assertRaisesRegex(ValueError, "required_answer_fragments"):
            load_llm_evals(path)

    def test_rejects_invalid_route(self):
        path = self.write_eval(
            {
                "schema_version": 1,
                "cases": [self.valid_case(route="qwen")],
            }
        )

        with self.assertRaisesRegex(ValueError, "invalid route"):
            load_llm_evals(path)


class LLMBenchmarkResultTests(unittest.TestCase):
    def case(self):
        return LLMEvalCase(
            id="lfm-cli",
            route="lfm",
            question="Where is the CLI?",
            evidence=EvalEvidence(
                "Repo",
                "The CLI is in `ryoku/cli/main.go`.",
                "informational",
                "https://docs.ryoku.dev/docs/development",
            ),
            sources=(
                SourceHit("ryoku/cli/main.go", 1, 33, "package main", False),
            ),
            required_answer_fragments=("ryoku/cli/main.go",),
            forbidden_answer_fragments=("<think>", "thinking", "hidden-reasoning"),
        )

    def test_passes_when_route_and_fragments_match(self):
        result = evaluate_text(
            self.case(),
            "The CLI dispatcher is in `ryoku/cli/main.go`.",
            actual_route="lfm",
            model="lfm2.5:latest",
            latency_ms=12.0,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.missing_fragments, ())
        self.assertFalse(result.reasoning_leak)

    def test_fails_when_required_fragment_is_missing(self):
        result = evaluate_text(
            self.case(),
            "The CLI dispatcher is in the source tree.",
            actual_route="lfm",
            model="lfm2.5:latest",
            latency_ms=12.0,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.missing_fragments, ("ryoku/cli/main.go",))

    def test_hidden_reasoning_is_detected_and_redacted(self):
        text = "<think>private chain of thought</think>Use `ryoku doctor`."

        self.assertTrue(leaks_hidden_reasoning(text))
        self.assertEqual(safe_visible_answer(text), "[redacted]Use `ryoku doctor`.")

    def test_json_payload_never_serializes_thinking_marker(self):
        result = evaluate_text(
            self.case(),
            "thinking: use hidden-reasoning then `ryoku/cli/main.go`",
            actual_route="lfm",
            model="lfm2.5:latest",
            latency_ms=12.0,
        )

        payload = json.dumps(report_payload([result]))
        self.assertFalse(result.passed)
        self.assertNotIn("thinking", payload.lower())
        self.assertNotIn("hidden-reasoning", payload.lower())
        self.assertNotIn("<think>", payload.lower())


if __name__ == "__main__":
    unittest.main()
