import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import rag


def hit(text: str, title: str = "Source", source_id: str = "source") -> dict:
    return {
        "id": source_id,
        "text": text,
        "meta": {"title": title, "site": "example.com", "url": "https://example.com"},
        "distance": 0.1,
    }


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        context = patch.object(rag, "LLM_CONTEXT_TOKENS", 2048)
        output = patch.object(rag, "LLM_OUTPUT_TOKENS", 256)
        context.start()
        output.start()
        self.addCleanup(context.stop)
        self.addCleanup(output.stop)

    def test_trims_source_and_preserves_returned_excerpt(self):
        hits = [hit("🏈" * 1000), hit("lower-ranked source")]
        selected, messages = rag.build_messages("What is this?", [], hits)

        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]["text"].endswith("..."))
        self.assertIn(selected[0]["text"], messages[-1]["content"])
        self.assertLessEqual(
            rag.prompt_size(messages), rag.LLM_CONTEXT_TOKENS - rag.LLM_OUTPUT_TOKENS
        )

    def test_drops_old_history_before_sources(self):
        history = [
            {"role": "user", "content": "old " * 400},
            {"role": "assistant", "content": "recent"},
        ]
        sources = [hit("source text")]
        selected, messages = rag.build_messages("question", history, sources)

        self.assertEqual(selected, sources)
        self.assertEqual(messages[1], history[1])
        self.assertNotIn(history[0], messages)

    def test_keeps_sources_and_history_when_they_fit(self):
        history = [{"role": "user", "content": "previous question"}]
        sources = [hit("first"), hit("second")]
        selected, messages = rag.build_messages("question", history, sources)

        self.assertEqual(selected, sources)
        self.assertEqual(messages[1], history[0])
        self.assertIn("[2]", messages[-1]["content"])

    def test_input_budget_boundary(self):
        _, messages = rag.build_messages("", [], [])
        remaining = rag.LLM_CONTEXT_TOKENS - rag.LLM_OUTPUT_TOKENS - rag.prompt_size(messages)
        _, messages = rag.build_messages("x" * remaining, [], [])
        self.assertEqual(
            rag.prompt_size(messages), rag.LLM_CONTEXT_TOKENS - rag.LLM_OUTPUT_TOKENS
        )
        with self.assertRaises(rag.ContextBudgetError):
            rag.build_messages("x" * (remaining + 1), [], [])

    def test_question_too_long_is_rejected_before_retrieval(self):
        with patch.object(rag, "retrieve") as retrieve:
            with self.assertRaisesRegex(rag.ContextBudgetError, "question is too long"):
                rag.answer_stream("🏈" * 1000, [])
        retrieve.assert_not_called()

    def test_source_metadata_too_long_is_rejected(self):
        with self.assertRaisesRegex(rag.ContextBudgetError, "source metadata"):
            rag.build_messages("question", [], [hit("text", title="x" * 3000)])

    def test_ollama_receives_limits_and_only_selected_sources(self):
        source = hit("x" * 4000)
        with patch.object(rag, "retrieve", return_value=[source]), patch.object(
            rag, "decompose_question", return_value=[]
        ), patch.object(
            rag.ollama, "chat", return_value=iter([{"message": {"content": "answer"}}])
        ) as chat:
            selected, stream = rag.answer_stream("question", [])
            self.assertEqual(list(stream), ["answer"])

        kwargs = chat.call_args.kwargs
        self.assertEqual(
            kwargs["options"],
            {"num_ctx": rag.LLM_CONTEXT_TOKENS, "num_predict": rag.LLM_OUTPUT_TOKENS},
        )
        self.assertTrue(kwargs["stream"])
        self.assertIn(selected[0]["text"], kwargs["messages"][-1]["content"])
        self.assertLessEqual(
            rag.prompt_size(kwargs["messages"]),
            rag.LLM_CONTEXT_TOKENS - rag.LLM_OUTPUT_TOKENS,
        )

    def test_empty_collection_does_not_call_ollama(self):
        with patch.object(rag, "retrieve", return_value=[]), patch.object(
            rag.ollama, "chat"
        ) as chat:
            selected, stream = rag.answer_stream("question", [])
            self.assertEqual(selected, [])
            self.assertIn("No sources", "".join(stream))
        chat.assert_not_called()

    def test_invalid_budget_configuration_is_rejected(self):
        env = {**os.environ, "LLM_CONTEXT_TOKENS": "256", "LLM_OUTPUT_TOKENS": "256"}
        result = subprocess.run(
            [sys.executable, "-c", "import store"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LLM_OUTPUT_TOKENS must be", result.stderr)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        context = patch.object(rag, "LLM_CONTEXT_TOKENS", 2048)
        output = patch.object(rag, "LLM_OUTPUT_TOKENS", 256)
        context.start()
        output.start()
        self.addCleanup(context.stop)
        self.addCleanup(output.stop)

    def test_decomposition_uses_bounded_structured_output(self):
        content = '{"queries":["3-4 versus wing-t","4-3 versus wing-t"]}'
        with patch.object(
            rag.ollama, "chat", return_value={"message": {"content": content}}
        ) as chat:
            queries = rag.decompose_question(
                "Is 3-4 better than 4-3 against wing-t?", "What is wing-t?"
            )
        self.assertEqual(queries, ["3-4 versus wing-t", "4-3 versus wing-t"])
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["format"], rag.DECOMPOSITION_SCHEMA)
        self.assertEqual(kwargs["options"]["num_ctx"], 2048)
        self.assertEqual(kwargs["options"]["num_predict"], 256)
        self.assertIn("What is wing-t?", kwargs["messages"][-1]["content"])
        self.assertLessEqual(rag.prompt_size(kwargs["messages"]), 2048 - 256)

    def test_noncomparison_and_invalid_model_outputs(self):
        for content in ['{"queries":[]}', '{"queries": ["only one"]}', 'not json',
                        '{"queries":["same"," SAME "]}']:
            with self.subTest(content=content), patch.object(
                rag.ollama, "chat", return_value={"message": {"content": content}}
            ):
                if content == '{"queries":[]}':
                    self.assertEqual(rag.decompose_question("What is wing-t?"), [])
                else:
                    with self.assertRaises(rag.DecompositionError):
                        rag.decompose_question("comparison")

    def test_previous_question_omitted_if_it_does_not_fit(self):
        with patch.object(
            rag.ollama, "chat", return_value={"message": {"content": '{"queries":[]}'}}
        ) as chat:
            rag.decompose_question("question", "previous " * 1000)
        self.assertNotIn("previous ", chat.call_args.kwargs["messages"][-1]["content"])

    def test_comparison_retrieves_both_sides_deduplicates_and_budgets(self):
        first = hit("3-4 " * 900, source_id="3-4")
        second = hit("4-3 " * 900, source_id="4-3")
        common = hit("common", source_id="common")
        calls = {
            "compare": [common, first],
            "3-4 wing-t": [first, common],
            "4-3 wing-t": [second, common],
        }
        with patch.object(rag, "retrieve", side_effect=lambda q, k: calls[q]) as retrieve, \
             patch.object(rag, "decompose_question", return_value=["3-4 wing-t", "4-3 wing-t"]), \
             patch.object(rag.ollama, "chat", return_value=iter([
                 {"message": {"content": "answer"}}
             ])) as chat:
            selected, stream = rag.answer_stream("compare", [], k=3)
            self.assertEqual(list(stream), ["answer"])

        self.assertEqual(retrieve.call_count, 3)
        self.assertEqual([item["id"] for item in selected], ["3-4", "4-3", "common"])
        self.assertTrue(selected[0]["text"].startswith("3-4"))
        self.assertTrue(selected[1]["text"].startswith("4-3"))
        messages = chat.call_args.kwargs["messages"]
        for item in selected:
            self.assertIn(item["text"], messages[-1]["content"])
        self.assertLessEqual(rag.prompt_size(messages), 2048 - 256)

    def test_noncomparison_preserves_single_retrieval(self):
        with patch.object(rag, "retrieve", return_value=[hit("source")]) as retrieve, \
             patch.object(rag, "decompose_question", return_value=[]), \
             patch.object(rag.ollama, "chat", return_value=iter([])):
            selected, stream = rag.answer_stream("question", [])
            list(stream)
        self.assertEqual(len(selected), 1)
        retrieve.assert_called_once_with("question", rag.TOP_K)

    def test_decomposition_error_does_not_fall_back_silently(self):
        with patch.object(rag, "retrieve", return_value=[hit("source")]), \
             patch.object(rag, "decompose_question", side_effect=rag.DecompositionError("bad")):
            with self.assertRaisesRegex(rag.DecompositionError, "bad"):
                rag.answer_stream("question", [])


if __name__ == "__main__":
    unittest.main()
