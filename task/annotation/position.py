"""
Position annotation task: height comparison & proximity.

Sub-tasks:
    height_comparison — compare the vertical position (z-max) of two objects,
                        randomly ask "which is higher/lower" in OE or MCQ form.
    proximity         — classify object pairs as "next to" or "far away" based on
                        the ratio of point-cloud distance to average object extent,
                        then generate one QA per category.

Templates used:
    position.height_higher  — [O] options, [X] answer
    position.height_lower   — [O] options, [X] answer
    position.next_far       — [A]/[B] object names, [O] options, [X] answer
"""

import random
import numpy as np
from itertools import combinations
from .core.base_annotation_task import BaseAnnotationTask
from .core.visual_marker import MarkConfig
from utils.box_utils import compute_box_3d_corners
from .core.question_type import QuestionType

from utils.point_cloud_utils import compute_point_cloud_distance


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Position"
    SUB_TASKS = {
        "height_comparison": {"default": 1, "handler": "_generate_height_comparison"},
        "proximity":         {"default": 1, "handler": "_generate_proximity"},
    }

    def get_mark_config(self):
        return MarkConfig(mark_types=["mask", "box"])

    # ── helpers ──────────────────────────────────────────────────────

    def _get_z_max(self, node):
        """Compute the maximum Z coordinate from the 3D bounding box corners."""
        box_3d_world = node.box_3d_world
        corners = compute_box_3d_corners(np.array(box_3d_world[:3]), np.array(box_3d_world[3:6]), box_3d_world[6:])
        return np.max(corners[:, -1])

    # ── prompt functions (called by mark_and_prompt) ─────────────────

    def height_comparison_prompt_func(self, A, B, question_type=QuestionType.OPEN_ENDED):
        """Generate a height comparison QA for two marked objects.

        Randomly picks higher/lower template. Format (OE/MCQ) is controlled
        by the question_type parameter passed from the handler.
        """
        A_desc, A_node = A
        B_desc, B_node = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        is_above = self._get_z_max(A_node) > self._get_z_max(B_node)

        # randomly ask "higher" or "lower"
        if random.random() < 0.5:
            tpl_name = "position.height_higher"
            target = A_desc if is_above else B_desc
        else:
            tpl_name = "position.height_lower"
            target = A_desc if not is_above else B_desc

        if question_type == QuestionType.OPEN_ENDED:
            options = f"The {A_desc} or the {B_desc}?"
            return self.render_prompt(tpl_name, shared={"O": options, "X": target})
        else:
            options = f"\nOptions: A:The {A_desc} B:The {B_desc}"
            answer = "A" if target == A_desc else "B"
            return self.render_prompt(tpl_name, shared={"O": options, "X": answer})

    def proximity_prompt_func(self, A, B, next_or_far):
        """Generate a proximity MCQ for two marked objects.

        Options are shuffled so "next to" / "far away" appear in random order.
        """
        A_desc, A_node = A
        B_desc, B_node = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        # shuffle option order
        next_far = ["next to each other", "far away from each other"]
        if random.random() < 0.5:
            next_far = next_far[::-1]

        options = f"\nOptions: A: {next_far[0]} B: {next_far[1]}"

        if next_or_far == "next":
            answer = "A" if next_far[0] == "next to each other" else "B"
        else:
            answer = "A" if next_far[0] == "far away from each other" else "B"

        return self.render_prompt("position.next_far", shared={"A": A_desc, "B": B_desc, "O": options, "X": answer})

    # ── handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_height_comparison(self, graph):
        """Sample two objects and ask which is higher/lower (50% OE / 50% MCQ)."""
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) < 2:
            return None
        image = graph.primary_view.image
        sampled = random.sample(nodes, 2)

        # decide format before calling prompt func so QuestionType stays in sync
        if random.random() < 0.5:
            qtype = QuestionType.OPEN_ENDED
            prompt_args = {"question_type": QuestionType.OPEN_ENDED}
        else:
            qtype = QuestionType.MCQ
            prompt_args = {"question_type": QuestionType.MCQ}

        prompt, processed_image = self.mark_and_prompt(
            sampled, image, self.height_comparison_prompt_func,
            mark_prob=0.8, prompt_args=prompt_args,
        )
        cog_ctx = self._make_singleview_cog_context(graph, sampled)
        return prompt, processed_image, qtype, cog_ctx

    def _generate_proximity(self, graph):
        """Classify all object pairs into "next" / "far" by size-relative
        distance ratio, then sample one pair from each category.

        Ratio = point_cloud_min_distance / avg_object_extent
            < 0.5  → "next to each other"
            > 2.0  → "far away from each other"

        Returns list[(prompt, image, qtype)], at most 2 items (one next, one far).
        """
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) < 2:
            return None
        # Cap to avoid O(n²) blowup in pair enumeration
        if len(nodes) > 8:
            nodes = random.sample(nodes, 8)
        image = graph.primary_view.image

        # classify pairs by distance / avg_extent ratio
        next_candidates, far_candidates = [], []
        for nodeA, nodeB in combinations(nodes, 2):
            if nodeA.tag == nodeB.tag:
                continue
            pcA = nodeA.view_appearances[0].pointcloud_camera
            pcB = nodeB.view_appearances[0].pointcloud_camera
            distance = compute_point_cloud_distance(pcA, pcB)
            avg_extent = np.mean([
                np.mean(nodeA.box_3d_world[3:6]),
                np.mean(nodeB.box_3d_world[3:6]),
            ])
            ratio = distance / avg_extent if avg_extent > 0 else float('inf')
            if ratio < 0.5:
                next_candidates.append((nodeA, nodeB))
            elif ratio > 2.0:
                far_candidates.append((nodeA, nodeB))

        if not next_candidates and not far_candidates:
            return None

        # one QA per category
        results = []
        if next_candidates:
            pair = random.choice(next_candidates)
            prompt, img = self.mark_and_prompt(
                pair, image, self.proximity_prompt_func,
                mark_prob=0.5, prompt_args={"next_or_far": "next"}
            )
            cog_ctx = self._make_singleview_cog_context(graph, list(pair))
            results.append((prompt, img, QuestionType.MCQ, cog_ctx))
        if far_candidates:
            pair = random.choice(far_candidates)
            prompt, img = self.mark_and_prompt(
                pair, image, self.proximity_prompt_func,
                mark_prob=0.5, prompt_args={"next_or_far": "far"}
            )
            cog_ctx = self._make_singleview_cog_context(graph, list(pair))
            results.append((prompt, img, QuestionType.MCQ, cog_ctx))

        return results
