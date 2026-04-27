"""
Object counting annotation task: count duplicate objects in a scene.

Sub-tasks:
    count_oe  — how many X are there? (open-ended, answer is a number)
    count_mcq — same question with 4 numeric options (A/B/C/D)

Templates used:
    counting.open_ended — [A] object name, [X] count
    counting.mcq        — [X] object name (q) / answer letter (a), [Y] options
"""

import random
from collections import Counter
from .core.base_annotation_task import BaseAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Counting"
    SUB_TASKS = {
        "count_oe":  {"default": 1, "handler": "_generate_count_oe"},
        "count_mcq": {"default": 1, "handler": "_generate_count_mcq"},
    }

    def check_example(self, example) -> bool:
        if not super().check_example(example):
            return False
        return len(example["obj_tags"]) > 1

    # ── helpers ──────────────────────────────────────────────────────

    def _get_tag_counts(self, graph):
        """Return {tag: count} for tags appearing more than once, respecting filter_tags."""
        if self.filter_tags is None:
            return graph.duplicate_tags
        tags = [t for t in graph.obj_tags if t not in self.filter_tags]
        return {tag: cnt for tag, cnt in Counter(tags).items() if cnt > 1}

    @staticmethod
    def _generate_wrong_counts(correct, num=3, range_diff=5):
        """Generate `num` distinct wrong counts near `correct`."""
        lo = max(0, correct - range_diff)
        hi = correct + range_diff
        wrong = set()
        for _ in range(100):
            n = random.randint(lo, hi)
            if n != correct:
                wrong.add(n)
                if len(wrong) == num:
                    break
        return list(wrong)

    # ── prompt functions ─────────────────────────────────────────────

    def count_oe_prompt_func(self, tag, count):
        """Generate an open-ended counting QA."""
        plural = tag + "s" if count > 1 else tag
        return self.render_prompt(
            "counting.open_ended",
            shared={"X": str(count)},
            q_args={"A": tag},
            a_args={"A": plural},
        )

    def count_mcq_prompt_func(self, tag, count):
        """Generate an MCQ counting QA with 4 options."""
        wrong = self._generate_wrong_counts(count)
        if len(wrong) < 3:
            return None
        candidates = [count] + wrong
        shuffled, answer = self._shuffle_mcq(candidates)
        options = [f"{'ABCD'[i]}:{shuffled[i]}" for i in range(4)]
        return self.render_prompt(
            "counting.mcq",
            shared={"Y": "\n".join(options)},
            q_args={"X": tag},
            a_args={"X": answer},
        )

    # ── handlers ─────────────────────────────────────────────────────

    def _generate_count_oe(self, graph):
        """Pick a random duplicate tag and ask how many there are."""
        tag_counts = self._get_tag_counts(graph)
        if not tag_counts:
            return None
        tag = random.choice(list(tag_counts.keys()))
        prompt = self.count_oe_prompt_func(tag, tag_counts[tag])
        image_bytes = {"bytes": convert_pil_to_bytes(graph.primary_view.image)}
        nodes = [n for n in graph.get_object_nodes() if n.tag == tag]
        cog_ctx = self._make_singleview_cog_context(graph, nodes)
        return prompt, image_bytes, QuestionType.OPEN_ENDED, cog_ctx

    def _generate_count_mcq(self, graph):
        """Pick a random duplicate tag and ask how many (MCQ)."""
        tag_counts = self._get_tag_counts(graph)
        if not tag_counts:
            return None
        tag = random.choice(list(tag_counts.keys()))
        prompt = self.count_mcq_prompt_func(tag, tag_counts[tag])
        if prompt is None:
            return None
        image_bytes = {"bytes": convert_pil_to_bytes(graph.primary_view.image)}
        nodes = [n for n in graph.get_object_nodes() if n.tag == tag]
        cog_ctx = self._make_singleview_cog_context(graph, nodes)
        return prompt, image_bytes, QuestionType.MCQ, cog_ctx
