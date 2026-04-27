"""
Size annotation task: absolute size measurement & relative size comparison.

Sub-tasks:
    absolute_size  — measure the longest extent or height of an object in metres /
                     centimetres (requires metric depth & 3D bounding boxes).
    relative_size  — compare the volume of two objects and ask which is bigger /
                     smaller.

Templates used:
    size.big.single_view       — condition-based, [A]/[B] object names
    size.small.single_view     — condition-based, [A]/[B] object names
    size.absolute.single_view  — [A] object name, [X] value+unit, [D] disclaimer
    size.height.single_view    — [A] object name, [X] value+unit, [D] disclaimer
"""

import random
from ..prompt_templates.size_prompt_templates import (
    unit_centimeter_disclaimer, unit_meter_disclaimer,
)
from .core.base_annotation_task import BaseAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Size"
    SUB_TASKS = {
        "absolute_size":   {"default": 1, "handler": "_generate_absolute_size"},
        "relative_size":   {"default": 1, "handler": "_generate_relative_size"},
    }

    def get_mark_config(self):
        return MarkConfig(mark_types=["mask", "box", "point"], shuffle_colors=True)

    # ── helpers ──────────────────────────────────────────────────────

    def _get_node_extent(self, node):
        """Get the 3D bounding box extent (xl, yl, zl) of a node.

        Prefers box_3d_world dimensions; falls back to axis-aligned
        bounding box of the camera-space pointcloud.
        """
        if node.box_3d_world is not None:
            return node.box_3d_world[3:6]
        cloud = node.view_appearances[0].pointcloud_camera
        return cloud.get_axis_aligned_bounding_box().get_extent()

    # ── prompt functions (called by mark_and_prompt or handler) ─────

    def relative_size_prompt_func(self, A, B):
        """Generate a relative size QA for two marked objects.

        Computes volume from 3D extent, randomly asks "bigger" or "smaller",
        uses condition-based template (True branch = A satisfies the condition).
        """
        A_desc, A_node = A
        B_desc, B_node = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        ext_A = self._get_node_extent(A_node)
        ext_B = self._get_node_extent(B_node)
        volume_A = ext_A[0] * ext_A[1] * ext_A[2]
        volume_B = ext_B[0] * ext_B[1] * ext_B[2]

        # randomly ask "bigger" or "smaller"
        if random.random() < 0.5:
            return self.render_prompt("size.big.single_view", condition=volume_A > volume_B, shared={"A": A_desc, "B": B_desc})
        else:
            return self.render_prompt("size.small.single_view", condition=volume_A < volume_B, shared={"A": A_desc, "B": B_desc})

    def absolute_size_prompt_func(self, marked, template_name, get_value):
        """Generate an absolute size QA for a single marked object.

        Args:
            marked:        (desc, node) from marker output.
            template_name: which template to use (absolute / height).
            get_value:     callable(node) → float, extracts the metric value
                           (e.g. max extent or z-size) in metres.
        """
        desc, node = marked
        A_desc = desc.lower()
        value = get_value(node)

        # randomly pick unit and attach the corresponding disclaimer
        unit = random.choice(["cm", "m"])
        if unit == "cm":
            value *= 100
            disclaimer = random.choice(unit_centimeter_disclaimer)
        else:
            disclaimer = random.choice(unit_meter_disclaimer)

        return self.render_prompt(
            template_name, shared={"A": A_desc, "X": f"{round(value, 2)} {unit}", "D": disclaimer},
        )

    # ── handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_absolute_size(self, graph):
        """Measure absolute size / height of 1–4 objects.

        Requires metric depth data and 3D bounding boxes.
        For each marked object, generates two prompts:
          1) overall size (max extent dimension)
          2) height (z-size from box_3d_world)
        """
        if not graph.is_metric_depth:
            return None
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) == 0:
            return None
        image = graph.primary_view.image
        num = random.randint(2, min(len(nodes), 4)) if len(nodes) > 1 else 1
        sampled = random.sample(nodes, num)

        processed_image, marked = self.marker.mark_objects(image, sampled)

        prompts = []
        for m in marked:
            # overall size: longest extent dimension
            prompts.append(self.absolute_size_prompt_func(
                m, "size.absolute.single_view",
                lambda n: max(self._get_node_extent(n)),
            ))
            # height: z-dimension of the 3D bounding box
            prompts.append(self.absolute_size_prompt_func(
                m, "size.height.single_view",
                lambda n: n.box_3d_world[5],
            ))

        return prompts, processed_image, QuestionType.OPEN_ENDED, self._make_singleview_cog_context(graph, sampled)

    def _generate_relative_size(self, graph):
        """Compare the volume of two objects and ask which is bigger/smaller.

        Marks both objects with 70% probability; without marks the question
        relies on object names only.
        """
        nodes = [n for n in graph.get_object_nodes() if n.box_3d_world is not None]
        if len(nodes) < 2:
            return None
        image = graph.primary_view.image
        sampled = random.sample(nodes, 2)
        prompt, processed_image = self.mark_and_prompt(
            sampled, image, self.relative_size_prompt_func, mark_prob=0.7
        )
        cog_ctx = self._make_singleview_cog_context(graph, sampled)
        return prompt, processed_image, QuestionType.OPEN_ENDED, cog_ctx
