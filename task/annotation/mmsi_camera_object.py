"""
MMSI-Bench: Camera–Object direction task.

Given two diverse views and an object visible only in view B, asks where the
object is located relative to view-A's camera (in view-A's camera frame).

Answer space: 4 options among Front / Back / Left / Right.
"""

import random
import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Camera-Object"
    SUB_TASKS = {
        "camera_object_mcq": {"default": 1, "handler": "_generate_camera_object_mcq"},
    }

    # ─── Handler ─────────────────────────────────────────────────────

    def _find_view_b_only_node(self, graph, retries=30):
        """Pick (v_a, v_b, node) such that node is visible only in v_b.

        "Only in v_b" means node.view_appearances does not contain v_a — it
        may still be visible in other views besides v_b. Adds a pose-diversity
        check between v_a and v_b.
        """
        view_ids = [vi for vi in graph.views if graph.views[vi].pose is not None]
        if len(view_ids) < 2:
            return None
        nodes = [n for n in graph.nodes.values() if n.box_3d_world is not None
                 and n.tag not in ("floor", "ceiling", "wall")]
        if not nodes:
            return None

        for _ in range(retries):
            v_a = random.choice(view_ids)
            pose_a = graph.views[v_a].pose
            random.shuffle(nodes)
            for node in nodes:
                # Candidate "target" views where the node appears.
                cand_v_b = [v for v in node.view_appearances if v != v_a]
                if not cand_v_b:
                    continue
                # Ensure the node is not visible in v_a.
                if v_a in node.view_appearances:
                    continue
                v_b = random.choice(cand_v_b)
                pose_b = graph.views[v_b].pose
                if self._check_pose_diversity(pose_b, [pose_a],
                                              self.min_rot_angle,
                                              self.min_translation):
                    return v_a, v_b, node
        return None

    def _generate_camera_object_mcq(self, graph):
        result = self._find_view_b_only_node(graph)
        if result is None:
            return None
        v_a, v_b, node = result

        pose_a = np.asarray(graph.views[v_a].pose, dtype=float)
        center_world = np.array([*node.box_3d_world[:3], 1.0], dtype=float)
        center_in_a = np.linalg.inv(pose_a) @ center_world
        x = float(center_in_a[0])
        z = float(center_in_a[2])

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
            f"In image 2, where is the {node.tag} relative to the camera that "
            "took image 1? " + options_str
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
