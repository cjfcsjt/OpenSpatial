"""
Manipulation-like task (viewpoint-difference approximation, all-angles style).

Since we have no true "object was manipulated" data, this task produces the
closest analogue: picks two views sharing a common anchor object, and asks
what apparent change happened between the two viewpoints. The question text
explicitly notes this is a camera-viewpoint difference, not a physical
manipulation.

Answer space (4-option MCQ):
    A. Rotated ~90° (clockwise or counter-clockwise depending on sign)
    B. Translated to the left (or right)
    C. Tilted
    D. Unchanged
"""

import math
import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


ROT_THRESHOLD_DEG = 60.0
TRANS_THRESHOLD_M = 0.3
TILT_THRESHOLD_DEG = 20.0


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Manipulation Viewpoint"
    SUB_TASKS = {
        "manipulation_view_mcq": {"default": 1, "handler": "_generate_manipulation_view_mcq"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.rot_threshold_deg = args.get("rot_threshold_deg", ROT_THRESHOLD_DEG)
        self.trans_threshold_m = args.get("trans_threshold_m", TRANS_THRESHOLD_M)
        self.tilt_threshold_deg = args.get("tilt_threshold_deg", TILT_THRESHOLD_DEG)

    # ─── Handler ─────────────────────────────────────────────────────

    def _generate_manipulation_view_mcq(self, graph):
        node, views = self._find_overlapping_views(graph, num_views=2)
        if node is None or node.box_3d_world is None:
            return None
        v_a, v_b = views
        pose_a = graph.views[v_a].pose
        pose_b = graph.views[v_b].pose
        if pose_a is None or pose_b is None:
            return None

        pose_a = np.asarray(pose_a, dtype=float)
        pose_b = np.asarray(pose_b, dtype=float)

        # Yaw around anchor (clockwise/counter-clockwise).
        anchor_xy = np.asarray(node.box_3d_world[:2], dtype=float)
        vec_a = pose_a[:3, 3][:2] - anchor_xy
        vec_b = pose_b[:3, 3][:2] - anchor_xy
        if np.linalg.norm(vec_a) > 1e-3 and np.linalg.norm(vec_b) > 1e-3:
            cos_theta = float(np.dot(vec_a, vec_b) /
                              (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
            cos_theta = max(-1.0, min(1.0, cos_theta))
            yaw_angle = math.degrees(math.acos(cos_theta))
            cross_z = float(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])
        else:
            yaw_angle = 0.0
            cross_z = 0.0

        # Lateral translation (in view-A camera frame, x axis).
        trans_world = pose_b[:3, 3] - pose_a[:3, 3]
        trans_cam = pose_a[:3, :3].T @ trans_world
        lateral = float(trans_cam[0])

        # Pitch / tilt: camera forward z-component change.
        forward_a = pose_a[:3, :3] @ np.array([0.0, 0.0, 1.0])
        forward_b = pose_b[:3, :3] @ np.array([0.0, 0.0, 1.0])
        pitch_angle = math.degrees(math.asin(
            max(-1.0, min(1.0, forward_b[2] - forward_a[2]))
        ))
        pitch_angle = abs(pitch_angle)

        if yaw_angle >= self.rot_threshold_deg:
            direction = "clockwise" if cross_z < 0 else "counter-clockwise"
            answer_text = f"Rotated ~90° {direction}"
        elif abs(lateral) >= self.trans_threshold_m:
            side = "left" if lateral < 0 else "right"
            answer_text = f"Translated to the {side}"
        elif pitch_angle >= self.tilt_threshold_deg:
            answer_text = "Tilted"
        else:
            answer_text = "Unchanged"

        # Fixed 4-option pool; randomize order.
        options = [
            "Rotated ~90° clockwise",
            "Translated to the left",
            "Tilted",
            "Unchanged",
        ]
        # Insert the ground-truth version of the answer if it differs
        # (e.g. counter-clockwise vs clockwise, right vs left).
        if answer_text not in options:
            # Replace one of the similar options.
            if answer_text.startswith("Rotated"):
                options[0] = answer_text
            elif answer_text.startswith("Translated"):
                options[1] = answer_text
            else:
                options[2] = answer_text
        random.shuffle(options)
        answer_letter = "ABCD"[options.index(answer_text)]
        options_str = "Options: " + " ".join(
            [f"{'ABCD'[i]}. {options[i]}" for i in range(4)]
        )
        question = (
            "Comparing View 1 and View 2 (which show a camera viewpoint "
            f"difference around the {node.tag}, not a physical object "
            "manipulation), what apparent change is observed? "
            + options_str
        )
        prompt = question + " Answer: " + answer_letter

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
