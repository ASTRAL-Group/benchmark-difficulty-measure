"""Tests for parsing model JSON into ProblemItem."""

from __future__ import annotations

import unittest

from generate.json_items import parse_problem_items_from_model_text


class TestParseProblemItems(unittest.TestCase):
    def test_array(self) -> None:
        text = '```json\n[{"id":"1","question":"Q?","reference_answer":"A","difficulty":0.5}]\n```'
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "1")
        self.assertEqual(items[0].question, "Q?")
        self.assertEqual(items[0].reference_answer, "A")
        self.assertEqual(items[0].difficulty, 0.5)

    def test_three_fields_only(self) -> None:
        text = '```json\n[{"id":"x","question":"q","reference_answer":"a"}]\n```'
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].difficulty)

    def test_answer_key(self) -> None:
        text = '```json\n[{"id":"x","question":"q","answer":"from_answer_key"}]\n```'
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].reference_answer, "from_answer_key")
        self.assertIsNone(items[0].modification_policy)

    def test_modification_policy(self) -> None:
        text = (
            '```json\n[{"id":"1","question":"Q","answer":"1","modification_policy":"small_edit"}]\n```'
        )
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].modification_policy, "small_edit")

    def test_fenced_json(self) -> None:
        text = '```json\n[{"id":"x","question":"q","reference_answer":"a","difficulty":null}]\n```'
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].difficulty)

    def test_preamble_then_array(self) -> None:
        text = (
            "Here is the JSON:\n```json\n"
            '[{"id":"1","question":"Q","reference_answer":"A"}]\n'
            "```"
        )
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "1")

    def test_invalid_json_latex_repaired_when_json_repair_installed(self) -> None:
        """Stdlib JSON rejects ``\\(`` in strings; json_repair fixes common LLM output."""
        try:
            import json_repair  # noqa: F401
        except ImportError:
            self.skipTest("json_repair not installed")
        text = r'''```json
[{"id":"1","question":"Define \( f(x)=x \)","reference_answer":"1"}]
```'''
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertIn("f", items[0].question)

    def test_latex_le_single_backslash_stdlib_fixed(self) -> None:
        """Models emit ``\\le`` with one backslash; must parse without relying on json_repair."""
        # JSON source contains invalid ``\l`` — not ``\\l`` (which would be valid).
        text = r'''```json
[{"id":"benchmark_item_1","question":"$1 \le k \le 50$","answer":"21"}]
```'''
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].reference_answer, "21")
        self.assertIn("le", items[0].question)

    def test_numeric_array_before_object_array(self) -> None:
        """Prose may contain ``[1,2,3]``; only the ```json block is parsed."""
        text = (
            "Example counts [1, 2, 3]. Final output:\n```json\n"
            '[{"id":"a","question":"q1","reference_answer":"1"},'
            '{"id":"b","question":"q2","reference_answer":"2"}]\n'
            "```"
        )
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].id, "a")
        self.assertEqual(items[1].reference_answer, "2")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_problem_items_from_model_text("")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_no_json_fence_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_problem_items_from_model_text('just prose without json')
        self.assertIn("parseable JSON array", str(ctx.exception))

    def test_loose_array_without_fence(self) -> None:
        text = 'Analysis…\n[{"id":"1","question":"Q","reference_answer":"A"}]'
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "1")

    def test_think_wrapper_stripped(self) -> None:
        text = (
            "<think>reasoning here</think>\n"
            "```JSON\n"
            '[{"id":"z","question":"q2","reference_answer":"a2"}]\n'
            "```"
        )
        items = parse_problem_items_from_model_text(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "z")


if __name__ == "__main__":
    unittest.main()
