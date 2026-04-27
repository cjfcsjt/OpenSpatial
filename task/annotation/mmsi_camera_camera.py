"""
MMSI-Bench: Camera–Camera direction task.

Given two diverse views, asks in which direction the second camera is located
relative to the first. Directions are computed in the first camera's
coordinate frame on the xz plane (forward = +z, right = +x).

Answer space: 4 options among Front / Back / Left / Right.
"""

import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Camera-Camera"
    SUB_TASKS = {
        "camera_camera_mcq": {"default": 1, "handler": "_generate_camera_camera_mcq"},
    }

    # ─── Handler ─────────────────────────────────────────────────────

    def _find_diverse_pair(self, graph, retries=10):
        """Pick 2 views whose poses differ enough (reuse _check_pose_diversity)."""
        view_ids = [vi for vi in graph.views if graph.views[vi].pose is not None]
        if len(view_ids) < 2:
            return None
        for _ in range(retries):
            v_a = random.choice(view_ids)
            pose_a = graph.views[v_a].pose
            shuffled = list(view_ids)
            random.shuffle(shuffled)
            for v_b in shuffled:
                if v_b == v_a:
                    continue
                pose_b = graph.views[v_b].pose
                if self._check_pose_diversity(pose_b, [pose_a],
                                              self.min_rot_angle,
                                              self.min_translation):
                    return v_a, v_b
        return None

    def _generate_camera_camera_mcq(self, graph):
        pair = self._find_diverse_pair(graph)
        if pair is None:
            return None
        v_a, v_b = pair

        pose_a = np.asarray(graph.views[v_a].pose, dtype=float)
        pose_b = np.asarray(graph.views[v_b].pose, dtype=float)

        # Express camera-B center in camera-A coordinates.
        cam_b_in_a = np.linalg.inv(pose_a) @ np.array([*pose_b[:3, 3], 1.0])
        x = float(cam_b_in_a[0])   # +x right
        z = float(cam_b_in_a[2])   # +z forward

        if abs(x) < 0.05 and abs(z) < 0.05:
            return None

        if abs(z) >= abs(x):
            answer_direction = "Front" if z > 0 else "Back"
        else:
            answer_direction = "Right" if x > 0 else "Left"

        options = ["Front", "Back", "Left", "Right"]
        random.shuffle(options)
        answer_letter = "ABCD"[options.index(answer_direction)]
        options_str = "Options: " + " ".join(
            [f"{'ABCD'[i]}. {options[i]}" for i in range(4)]
        )
        question = (
            "In image 2, the camera is located in which direction relative to "
            "image 1's camera? " + options_str
        )
        prompt = question + " Answer: " + answer_letter

        processed_images = [
            {"bytes": convert_pil_to_bytes(graph.views[v_a].image)},
            {"bytes": convert_pil_to_bytes(graph.views[v_b].image)},
        ]
        cog_ctx = self._make_cog_context(view_indices=[v_a, v_b])
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
