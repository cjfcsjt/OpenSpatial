"""
Camera clockwise annotation task (BLINK multi-view reasoning style).

Given two views sharing a common anchor object, asks whether the camera is
moving clockwise around the object (as seen from above).

Answer space: "A. Yes" / "B. No".
"""

import math
import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


MIN_ANGLE_DEG = 20.0   # Reject view pairs with small angular separation.


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Camera Clockwise"
    SUB_TASKS = {
        "clockwise_yes_no": {"default": 1, "handler": "_generate_clockwise_yes_no"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.min_angle_deg = args.get("min_angle_deg", MIN_ANGLE_DEG)

    # ─── Handler ─────────────────────────────────────────────────────

    def _generate_clockwise_yes_no(self, graph):
        node, views = self._find_overlapping_views(graph, num_views=2)
        if node is None or node.box_3d_world is None:
            return None
        if len(views) < 2:
            return None
        v_a, v_b = views[0], views[1]

        pose_a = graph.views[v_a].pose
        pose_b = graph.views[v_b].pose
        if pose_a is None or pose_b is None:
            return None

        anchor_xy = np.asarray(node.box_3d_world[:2], dtype=float)
        cam_a = np.asarray(pose_a[:3, 3][:2], dtype=float) - anchor_xy
        cam_b = np.asarray(pose_b[:3, 3][:2], dtype=float) - anchor_xy

        # Require non-degenerate 2D vectors.
        if np.linalg.norm(cam_a) < 1e-3 or np.linalg.norm(cam_b) < 1e-3:
            return None

        cos_theta = np.dot(cam_a, cam_b) / (np.linalg.norm(cam_a) * np.linalg.norm(cam_b))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = math.degrees(math.acos(cos_theta))
        if angle_deg < self.min_angle_deg:
            return None

        # z-component of 2D cross product = cam_a.x * cam_b.y - cam_a.y * cam_b.x
        cross_z = cam_a[0] * cam_b[1] - cam_a[1] * cam_b[0]
        # Going from A to B clockwise (when viewed from +Z looking down) means
        # cross_z < 0 in a right-handed coordinate frame.
        is_clockwise = cross_z < 0

        # Occasionally flip the question sense for diversity.
        ask_clockwise = random.random() < 0.5
        if ask_clockwise:
            question = (
                f"Is the camera moving clockwise around the {node.tag}?"
            )
            answer_truth = is_clockwise
        else:
            question = (
                f"Is the camera moving counter-clockwise around the {node.tag}?"
            )
            answer_truth = not is_clockwise

        candidates = ["Yes", "No"]
        correct_idx = 0 if answer_truth else 1
        options_str = "Options: A. Yes B. No"
        answer_letter = "A" if correct_idx == 0 else "B"

        prompt = question + " " + options_str + " Answer: " + answer_letter
        processed_images = [
            {"bytes": convert_pil_to_bytes(graph.views[v_a].image)},
            {"bytes": convert_pil_to_bytes(graph.views[v_b].image)},
        ]
        cog_ctx = self._make_cog_context(
            view_indices=[v_a, v_b],
            node_ids=[node.node_id],
            anchor_node_id=node.node_id,
        )
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
