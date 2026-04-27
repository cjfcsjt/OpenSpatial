"""
MMSI-Bench: Object-Object world-frame direction task.

Given two diverse views and two distinct objects visible in either view,
defines a "world frame" by taking the line between the two cameras as the
forward axis. Asks whether object-A is to the left or right of object-B in
this reference frame.

Answer space: "A. Left" / "B. Right" / "C. Same line" (within 0.3 m).
"""

import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


SAME_LINE_THRESHOLD = 0.3  # meters


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Object-Object"
    SUB_TASKS = {
        "object_object_mcq": {"default": 1, "handler": "_generate_object_object_mcq"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.same_line_threshold = args.get("same_line_threshold", SAME_LINE_THRESHOLD)

    # ─── Handler ─────────────────────────────────────────────────────

    def _find_two_nodes_two_views(self, graph, retries=30):
        view_ids = [vi for vi in graph.views if graph.views[vi].pose is not None]
        if len(view_ids) < 2:
            return None
        candidate_nodes = [n for n in graph.nodes.values()
                           if n.box_3d_world is not None
                           and n.tag not in ("floor", "ceiling", "wall")]
        if len(candidate_nodes) < 2:
            return None

        for _ in range(retries):
            v_a = random.choice(view_ids)
            pose_a = graph.views[v_a].pose
            v_b = random.choice(view_ids)
            if v_b == v_a:
                continue
            pose_b = graph.views[v_b].pose
            if not self._check_pose_diversity(pose_b, [pose_a],
                                              self.min_rot_angle,
                                              self.min_translation):
                continue
            pair = random.sample(candidate_nodes, 2)
            node_a, node_b = pair
            if node_a.tag == node_b.tag:
                continue
            # At least one of the two nodes must appear in v_a or v_b to be
            # observable from the question premise.
            reachable = (
                v_a in node_a.view_appearances or v_b in node_a.view_appearances
                or v_a in node_b.view_appearances or v_b in node_b.view_appearances
            )
            if not reachable:
                continue
            return v_a, v_b, node_a, node_b
        return None

    def _generate_object_object_mcq(self, graph):
        result = self._find_two_nodes_two_views(graph)
        if result is None:
            return None
        v_a, v_b, node_a, node_b = result

        cam_a = np.asarray(graph.views[v_a].pose[:3, 3], dtype=float)
        cam_b = np.asarray(graph.views[v_b].pose[:3, 3], dtype=float)
        forward = cam_b - cam_a
        forward_xy = forward[:2]
        if np.linalg.norm(forward_xy) < 1e-3:
            return None
        forward_xy = forward_xy / np.linalg.norm(forward_xy)
        # Right-hand perpendicular in xy plane: rotate forward 90° clockwise.
        right_xy = np.array([forward_xy[1], -forward_xy[0]])

        center_a_xy = np.asarray(node_a.box_3d_world[:2], dtype=float)
        center_b_xy = np.asarray(node_b.box_3d_world[:2], dtype=float)
        delta = center_a_xy - center_b_xy
        x_comp = float(np.dot(delta, right_xy))

        if abs(x_comp) < self.same_line_threshold:
            correct_idx = 2  # Same line
        elif x_comp > 0:
            correct_idx = 1  # A is to the right of B
        else:
            correct_idx = 0  # A is to the left of B

        candidates = ["Left", "Right", "Same line"]
        shuffled, answer_letter = self._shuffle_mcq(candidates, correct_idx)
        options_str = "Options: " + " ".join(
            [f"{'ABC'[i]}. {shuffled[i]}" for i in range(3)]
        )
        question = (
            f"Considering both images, is the {node_a.tag} to the left or "
            f"right of the {node_b.tag} in the world frame defined by the "
            "line connecting the two cameras? " + options_str
        )
        prompt = question + " Answer: " + answer_letter

        processed_images = [
            {"bytes": convert_pil_to_bytes(graph.views[v_a].image)},
            {"bytes": convert_pil_to_bytes(graph.views[v_b].image)},
        ]
        cog_ctx = self._make_cog_context(
            view_indices=[v_a, v_b],
            node_ids=[node_a.node_id, node_b.node_id],
        )
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
