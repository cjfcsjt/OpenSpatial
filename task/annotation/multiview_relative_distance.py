"""
Relative distance annotation task (all-angles style).

Given multiview SceneGraph data, picks 3 distinct objects (A/B/C) that all
appear in at least 2 views, and asks which of B/C is closer to A in 3D.

Answer: "A. Closer to B" / "B. Closer to C" / "C. Equal" (within 0.3 m).
"""

import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from .core.visual_marker import MarkConfig
from utils.image_utils import convert_pil_to_bytes


EQUAL_THRESHOLD = 0.3  # meters


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Relative Distance"
    SUB_TASKS = {
        "relative_distance_mcq": {"default": 1, "handler": "_generate_relative_distance_mcq"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.equal_threshold = args.get("equal_threshold", EQUAL_THRESHOLD)
        self.min_visible_views = args.get("min_visible_views", 2)

    def get_mark_config(self):
        return MarkConfig(mark_types=["mask", "box"])

    # ─── Data Finder ──────────────────────────────────────────────────

    def _find_triple_objects(self, graph):
        """Pick 3 objects with distinct tags, each visible in >= 2 views.

        Returns:
            (nodeA, nodeB, nodeC, views_to_show) or None.
        """
        candidate_nodes = []
        seen_tags = {}
        for nid, node in graph.nodes.items():
            if node.box_3d_world is None:
                continue
            if node.tag in ("floor", "ceiling", "wall"):
                continue
            if len(node.view_appearances) < self.min_visible_views:
                continue
            # Ensure each tag contributes at most one representative node
            if node.tag in seen_tags:
                continue
            seen_tags[node.tag] = nid
            candidate_nodes.append(node)

        if len(candidate_nodes) < 3:
            return None

        triple = random.sample(candidate_nodes, 3)
        # Pick up to 2 representative views showing any of the 3 objects.
        view_pool = set()
        for n in triple:
            view_pool.update(n.view_appearances.keys())
        if not view_pool:
            return None
        views_to_show = random.sample(
            list(view_pool), min(2, len(view_pool))
        )
        return triple[0], triple[1], triple[2], views_to_show

    # ─── Handler ─────────────────────────────────────────────────────

    def _generate_relative_distance_mcq(self, graph):
        result = self._find_triple_objects(graph)
        if result is None:
            return None
        nodeA, nodeB, nodeC, views = result

        # Compute 3D euclidean distances between box centers.
        cA = np.asarray(nodeA.box_3d_world[:3], dtype=float)
        cB = np.asarray(nodeB.box_3d_world[:3], dtype=float)
        cC = np.asarray(nodeC.box_3d_world[:3], dtype=float)
        dAB = float(np.linalg.norm(cA - cB))
        dAC = float(np.linalg.norm(cA - cC))

        if abs(dAB - dAC) < self.equal_threshold:
            correct_idx = 2  # Equal
        elif dAB < dAC:
            correct_idx = 0  # Closer to B
        else:
            correct_idx = 1  # Closer to C

        candidates = [
            f"Closer to {nodeB.tag}",
            f"Closer to {nodeC.tag}",
            "Equal",
        ]
        shuffled, answer_letter = self._shuffle_mcq(candidates, correct_idx)
        options_str = "Options: " + " ".join(
            [f"{'ABC'[i]}. {shuffled[i]}" for i in range(3)]
        )

        question = (
            f"Considering all views, is {nodeA.tag} closer to {nodeB.tag} "
            f"or to {nodeC.tag}? {options_str}"
        )
        prompt = question + " Answer: " + answer_letter

        processed_images = [
            {"bytes": convert_pil_to_bytes(graph.views[v].image)} for v in views
        ]
        cog_ctx = self._make_cog_context(
            view_indices=views,
            node_ids=[nodeA.node_id, nodeB.node_id, nodeC.node_id],
            anchor_node_id=nodeA.node_id,
        )
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
