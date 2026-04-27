"""
MMSI-Bench: Composite camera motion task.

Given two diverse views, decomposes the relative motion into:
  - Translation direction (Forward / Backward / Left / Right).
  - Rotation magnitude (geodesic angle between the two rotation matrices).

Produces a 4-option MCQ combining translation + rotation classes:
  A. Forward + left
  B. Backward + right
  C. Pure rotation
  D. Static

Thresholds: |translation| < 0.1 m is considered static; |rotation| ≥ 10°
combined with tiny translation yields "pure rotation".
"""

import math
import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


STATIC_TRANS_THR = 0.1    # meters
STATIC_ROT_THR_DEG = 10.0


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Camera Motion"
    SUB_TASKS = {
        "camera_motion_mcq": {"default": 1, "handler": "_generate_camera_motion_mcq"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.static_trans_thr = args.get("static_trans_thr", STATIC_TRANS_THR)
        self.static_rot_thr_deg = args.get("static_rot_thr_deg", STATIC_ROT_THR_DEG)

    # ─── Handler ─────────────────────────────────────────────────────

    def _find_pose_pair(self, graph, retries=10):
        view_ids = [vi for vi in graph.views if graph.views[vi].pose is not None]
        if len(view_ids) < 2:
            return None
        for _ in range(retries):
            v_a, v_b = random.sample(view_ids, 2)
            # Relax pose diversity check: we want all motion categories,
            # including "Static" and "Pure rotation" cases.
            return v_a, v_b
        return None

    def _generate_camera_motion_mcq(self, graph):
        pair = self._find_pose_pair(graph)
        if pair is None:
            return None
        v_a, v_b = pair

        pose_a = np.asarray(graph.views[v_a].pose, dtype=float)
        pose_b = np.asarray(graph.views[v_b].pose, dtype=float)

        # Translation in view-A's camera frame.
        trans_world = pose_b[:3, 3] - pose_a[:3, 3]
        trans_cam = pose_a[:3, :3].T @ trans_world
        trans_mag = float(np.linalg.norm(trans_cam))

        # Geodesic rotation angle between view-A and view-B.
        R_rel = pose_a[:3, :3].T @ pose_b[:3, :3]
        trace = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        rot_angle_deg = float(math.degrees(math.acos(trace)))

        # Translation classification (if non-static).
        if trans_mag < self.static_trans_thr:
            trans_class = None
            x, z = float(trans_cam[0]), float(trans_cam[2])
        else:
            x, z = float(trans_cam[0]), float(trans_cam[2])
            if abs(z) >= abs(x):
                trans_class = "forward" if z > 0 else "backward"
            else:
                trans_class = "right" if x > 0 else "left"

        # Build composite answer text + compose option set.
        if trans_class is None and rot_angle_deg < self.static_rot_thr_deg:
            answer_text = "Static"
        elif trans_class is None:
            answer_text = "Pure rotation"
        else:
            # Pick a plausible secondary axis name (lateral if main was
            # forward/backward, longitudinal otherwise) — this matches the
            # MMSI compound style "Forward + left".
            if trans_class in ("forward", "backward"):
                secondary = "left" if x < 0 else "right"
            else:
                secondary = "forward" if z > 0 else "backward"
            answer_text = f"{trans_class.capitalize()} + {secondary}"

        # 4 option pool: always present static/rotation as two fixed options;
        # pick two composite distractors to fill.
        composite_pool = [
            "Forward + left", "Forward + right",
            "Backward + left", "Backward + right",
        ]
        options = ["Static", "Pure rotation"]
        random.shuffle(composite_pool)
        options.extend(composite_pool[:2])
        # Make sure the correct answer is present.
        if answer_text not in options:
            options[2] = answer_text  # replace first composite distractor
        random.shuffle(options)
        answer_letter = "ABCD"[options.index(answer_text)]
        options_str = "Options: " + " ".join(
            [f"{'ABCD'[i]}. {options[i]}" for i in range(4)]
        )
        question = (
            "Between image 1 and image 2, the camera mainly moved... "
            + options_str
        )
        prompt = question + " Answer: " + answer_letter

        processed_images = [
            {"bytes": convert_pil_to_bytes(graph.views[v_a].image)},
            {"bytes": convert_pil_to_bytes(graph.views[v_b].image)},
        ]
        cog_ctx = self._make_cog_context(view_indices=[v_a, v_b])
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
