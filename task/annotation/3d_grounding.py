"""
3D grounding annotation task: predict 3D bounding boxes for objects.

Sub-tasks:
    grounding_oe — given object names, predict their 3D bounding boxes (open-ended).

Coordinate system:
    3D boxes are in camera coordinates, converted from world-frame boxes
    via SceneNode.box_3d_in_camera().

    Box format: [x, y, z, xl, yl, zl, roll, pitch, yaw]
    Euler angles in radians, rotation order zxy.

Templates used:
    grounding_3d.open_ended    — [A] object names, [X] box params json
    grounding_3d.camera_system — [H] hfov, [V] vfov, [W] width, [I] height
"""

import random
from .core.base_annotation_task import BaseAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes
from utils.projection_utils import compute_fov_from_intrinsic


class ThreeDGroundingGenerator(BaseAnnotationTask):

    QUESTION_TAG = "3D Grounding"
    SUB_TASKS = {
        "grounding_oe": {"default": 1, "handler": "_generate_grounding_oe"},
    }

    def check_example(self, example) -> bool:
        return super().check_example(example)

    # ── data preparation ─────────────────────────────────────────────

    def _store_camera_info(self, view):
        """Cache intrinsic and image dimensions in thread-local for message building."""
        depth_map = view.depth_map
        img_dim = depth_map.shape[::-1] if depth_map is not None else view.image.size
        self._thread_local.camera_info = (view.intrinsic, img_dim)

    # ── message builder override ─────────────────────────────────────

    def create_messages_from_prompts(self, prompts, processed_images=None):
        """Prepend camera system prompt to each question.

        Falls back to base behavior if camera info is unavailable.
        """
        camera_info = getattr(self._thread_local, 'camera_info', None)
        if camera_info is None:
            return super().create_messages_from_prompts(prompts, processed_images)

        intrinsic, img_dim = camera_info
        fov_h, fov_v = compute_fov_from_intrinsic(intrinsic, img_dim)

        tpl = self.get_template("grounding_3d.camera_system")
        sys_prompt, _ = tpl.render_qa(
            shared={"H": f"{fov_h:.2f}", "V": f"{fov_v:.2f}",
                    "W": str(img_dim[0]), "I": str(img_dim[1])},
        )

        messages = []
        for prompt in prompts:
            if "Answer: " not in prompt:
                continue
            question, answer = prompt.split("Answer: ", 1)
            question = sys_prompt + " <image> " + question.strip()
            messages.append([
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer.strip()},
            ])
        return messages

    # ── prompt function ────────────────────────────────────────────

    def grounding_oe_prompt_func(self, sampled_tags, tags_to_boxes):
        """Generate an open-ended 3D grounding QA.

        Args:
            sampled_tags: list of selected object tag strings.
            tags_to_boxes: dict mapping tag → list of camera-frame 9-param boxes.
        """
        box_params = [
            {"bbox_3d": [round(v, 2) for v in box], "label": tag}
            for tag in sampled_tags
            for box in tags_to_boxes[tag]
        ]
        return self.render_prompt(
            "grounding_3d.open_ended",
            shared={"A": ", ".join(sampled_tags), "X": str(box_params)},
        )

    # ── handler ───────────────────────────────────────────────────────

    def _generate_grounding_oe(self, graph):
        """Sample 1-3 object tags and ask for their 3D bounding boxes."""
        view = graph.primary_view
        pose = view.pose
        if pose is None:
            return None

        self._store_camera_info(view)

        nodes = graph.get_object_nodes()
        if self.filter_tags is not None:
            nodes = [n for n in nodes if n.tag not in self.filter_tags]

        tags_to_boxes = {}
        for node in nodes:
            cam_box = node.box_3d_in_camera(pose)
            if cam_box is not None:
                tags_to_boxes.setdefault(node.tag, []).append(cam_box)
        if not tags_to_boxes:
            return None

        unique_tags = list(tags_to_boxes.keys())
        sampled_tags = random.sample(unique_tags, random.randint(1, min(3, len(unique_tags))))

        prompt = self.grounding_oe_prompt_func(sampled_tags, tags_to_boxes)
        sampled_nodes = [n for n in nodes if n.tag in sampled_tags]
        cog_ctx = self._make_singleview_cog_context(graph, sampled_nodes)
        return prompt, {"bytes": convert_pil_to_bytes(view.image)}, QuestionType.OPEN_ENDED, cog_ctx
