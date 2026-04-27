"""
Camera movement direction annotation task (VSI-Bench style).

Picks a pair of views with the largest view-index gap and asks which dominant
direction the camera moved from start → end, measured in the start-frame
camera coordinate system.

Answer space: "A. Forward" / "B. Backward" / "C. Left" / "D. Right".
Minimum component magnitude 0.2 m; otherwise the sample is rejected.
"""

import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


MIN_COMPONENT_M = 0.2


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Camera Movement"
    SUB_TASKS = {
        "camera_movement_mcq": {"default": 1, "handler": "_generate_camera_movement_mcq"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.min_component_m = args.get("min_component_m", MIN_COMPONENT_M)

    # ─── Handler ─────────────────────────────────────────────────────

    def _generate_camera_movement_mcq(self, graph):
        # Collect all views whose pose is available.
        view_items = [(vi, graph.views[vi]) for vi in graph.views]
        view_items = [(vi, v) for (vi, v) in view_items if v.pose is not None]
        if len(view_items) < 2:
            return None

        # Pick first and last by view index (largest index gap).
        view_items.sort(key=lambda x: x[0])
        v_start_idx, view_start = view_items[0]
        v_end_idx, view_end = view_items[-1]

        pose_start = np.asarray(view_start.pose, dtype=float)
        pose_end = np.asarray(view_end.pose, dtype=float)

        t_start = pose_start[:3, 3]
        t_end = pose_end[:3, 3]
        delta_world = t_end - t_start

        # Transform delta into the start-frame camera coordinate system.
        # In this repo's convention (camera-to-world pose, +Z forward) we have
        # delta_cam = R_start^T @ (t_end - t_start). Forward = +z, right = +x,
        # down = +y.
        R_start = pose_start[:3, :3]
        delta_cam = R_start.T @ delta_world

        forward = float(delta_cam[2])     # +z in camera frame
        right = float(delta_cam[0])       # +x in camera frame
        # up/down ignored for this task.

        candidates = {
            "Forward": forward,
            "Backward": -forward,
            "Right": right,
            "Left": -right,
        }
        # Answer = direction with largest positive magnitude.
        answer_direction, answer_mag = max(candidates.items(), key=lambda kv: kv[1])
        if answer_mag < self.min_component_m:
            return None

        option_order = ["Forward", "Backward", "Left", "Right"]
        # Randomize option ordering for robustness.
        random.shuffle(option_order)
        answer_letter = "ABCD"[option_order.index(answer_direction)]
        options_str = "Options: " + " ".join(
            [f"{'ABCD'[i]}. {option_order[i]}" for i in range(4)]
        )
        question = (
            "From the start to the end of the video, the camera mainly "
            "moved in which direction? " + options_str
        )
        prompt = question + " Answer: " + answer_letter

        processed_images = [
            {"bytes": convert_pil_to_bytes(view_start.image)},
            {"bytes": convert_pil_to_bytes(view_end.image)},
        ]
        cog_ctx = self._make_cog_context(
            view_indices=[v_start_idx, v_end_idx],
        )
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
