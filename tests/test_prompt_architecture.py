"""Static invariants for the prompt architecture (zero LLM calls)."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from engine.llm import (
    EXTRACT_TAGS,
    PLAN_TAGS,
    REVIEW_TAGS,
    WRITE_TAGS,
    prompt_role_for_tag,
)


ROOT = Path(__file__).resolve().parent.parent


def _llm_calls():
    for folder in ("engine", "commands"):
        for path in sorted((ROOT / folder).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "call_llm"
                ):
                    yield path, node


class PromptArchitectureTests(unittest.TestCase):
    def test_every_call_has_an_explicit_tag(self):
        missing = [
            f"{path.relative_to(ROOT)}:{node.lineno}"
            for path, node in _llm_calls()
            if not any(keyword.arg == "tag" for keyword in node.keywords)
        ]
        self.assertEqual(missing, [], "untagged call_llm sites: " + ", ".join(missing))

    def test_literal_tags_have_a_registered_role(self):
        unknown = []
        for path, node in _llm_calls():
            tag_keyword = next(keyword for keyword in node.keywords if keyword.arg == "tag")
            if isinstance(tag_keyword.value, ast.Constant) and isinstance(tag_keyword.value.value, str):
                tag = tag_keyword.value.value
                if not prompt_role_for_tag(tag):
                    unknown.append(f"{path.relative_to(ROOT)}:{node.lineno}={tag!r}")
        self.assertEqual(unknown, [], "unregistered literal tags: " + ", ".join(unknown))

    def test_exact_role_tag_sets_are_disjoint(self):
        role_sets = {
            "planning": PLAN_TAGS,
            "writing": WRITE_TAGS,
            "extraction": EXTRACT_TAGS,
            "review": REVIEW_TAGS,
        }
        overlaps = []
        names = list(role_sets)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                shared = sorted(role_sets[left] & role_sets[right])
                if shared:
                    overlaps.append(f"{left}/{right}: {shared}")
        self.assertEqual(overlaps, [], "overlapping prompt roles: " + "; ".join(overlaps))


if __name__ == "__main__":
    unittest.main()
