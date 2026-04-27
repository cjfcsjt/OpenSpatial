"""
Distance annotation task: absolute distance & relative distance comparison.

Sub-tasks:
    absolute_distance — measure the point-cloud minimum distance between two objects
                        and report in metres or centimetres (requires metric depth).
    relative_distance — given three objects A, B, C, ask which of A/B is farther from /
                        closer to C, in both open-ended and MCQ form.

Templates used:
    distance.absolute_m     — [A]/[B] object names, [X] distance in metres
    distance.absolute_cm    — [A]/[B] object names, [X] distance in centimetres
    distance.relative_far   — [A]/[B]/[C] names, [X] answer, [O] options (empty for OE)
    distance.relative_close — [A]/[B]/[C] names, [X] answer, [O] options (empty for OE)
"""

import random
from .core.base_annotation_task import BaseAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType

from utils.point_cloud_utils import compute_point_cloud_distance


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Distance"
    SUB_TASKS = {
        "absolute_distance":  {"default": 1, "handler": "_generate_absolute_distance"},
        "relative_distance":  {"default": 1, "handler": "_generate_relative_distance"},
    }

    def get_mark_config(self):
        return MarkConfig(type_weights={"mask": 0.2, "box": 0.8})

    # ── helpers ──────────────────────────────────────────────────────

    def _get_cleaned_cloud(self, marked):
        """Extract and clean pointcloud from a marked result (desc, node).

        Returns (lowercased_desc, cleaned_pointcloud).
        """
        desc, cloud = self._get_cloud(marked)
        return desc.lower(), self._clean_cloud(cloud)

    def _resolve_relative_distance(self, A, B, C):
        """Compute relative distances and determine which is farther / closer.

        Given three marked objects A, B, C, computes min-distance from A→C
        and B→C, then returns descriptors and which is farther/closer to C.

        Returns:
            (A_desc, B_desc, C_desc, farther_desc, closer_desc,
             farther_tag, closer_tag)
            where tag is "A" or "B" (for MCQ option letters).
        """
        A_desc, A_cloud = self._get_cleaned_cloud(A)
        B_desc, B_cloud = self._get_cleaned_cloud(B)
        C_desc, C_cloud = self._get_cleaned_cloud(C)

        dist_AC = compute_point_cloud_distance(A_cloud, C_cloud)
        dist_BC = compute_point_cloud_distance(B_cloud, C_cloud)

        if dist_AC > dist_BC:
            farther, closer = A_desc, B_desc
            farther_tag, closer_tag = "A", "B"
        else:
            farther, closer = B_desc, A_desc
            farther_tag, closer_tag = "B", "A"

        return A_desc, B_desc, C_desc, farther, closer, farther_tag, closer_tag

    # ── prompt functions (called by mark_and_prompt or handler) ─────

    def absolute_distance_prompt_func(self, A, B):
        """Generate an absolute distance QA for two marked objects."""
        A_desc, A_cloud = self._get_cleaned_cloud(A)
        B_desc, B_cloud = self._get_cleaned_cloud(B)

        unit = random.choice(["m", "cm"])
        dist = compute_point_cloud_distance(A_cloud, B_cloud)
        scaled = dist * self.scaling_factor * (100 if unit == "cm" else 1)

        return self.render_prompt(
            f"distance.absolute_{unit}",
            shared={"A": A_desc, "B": B_desc, "X": f"{scaled:.2f} {unit}"},
        )

    def relative_distance_oe_prompt_func(self, A, B, C):
        """Generate an open-ended relative distance QA.

        Randomly asks "which is farther" or "which is closer" to C.
        [O] is set to empty string (no options for OE).
        """
        A_desc, B_desc, C_desc, farther, closer, _, _ = self._resolve_relative_distance(A, B, C)

        # randomly pick "farther" or "closer" question
        if random.random() < 0.5:
            return self.render_prompt("distance.relative_far", shared={"A": A_desc, "B": B_desc, "C": C_desc, "X": farther, "O": ""})
        else:
            return self.render_prompt("distance.relative_close", shared={"A": A_desc, "B": B_desc, "C": C_desc, "X": closer, "O": ""})

    def relative_distance_mcq_prompt_func(self, A, B, C):
        """Generate an MCQ relative distance QA.

        Builds "Options: A. <name>  B. <name>." and randomly asks farther/closer.
        Answer is the option letter (70% chance) or "letter. name" (30% chance).
        """
        A_desc, B_desc, C_desc, farther, closer, farther_tag, closer_tag = \
            self._resolve_relative_distance(A, B, C)

        options = f"\nOptions: A. {A_desc}  B. {B_desc}."

        if random.random() < 0.5:
            tpl_name, target_tag = "distance.relative_far", farther_tag
        else:
            tpl_name, target_tag = "distance.relative_close", closer_tag

        # 70% just letter, 30% letter + name for diversity
        answer = target_tag if random.random() < 0.7 else f"{target_tag}. {farther if target_tag == farther_tag else closer}"
        return self.render_prompt(tpl_name, shared={"A": A_desc, "B": B_desc, "C": C_desc, "X": answer, "O": options})

    # ── handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_absolute_distance(self, graph):
        """Measure the absolute distance between two objects.

        Requires metric depth data.
        Always marks both objects on the image (mark_prob=1.0).
        """
        if not graph.is_metric_depth:
            return None
        nodes = graph.get_object_nodes()
        if len(nodes) < 2:
            return None
        image = graph.primary_view.image
        sampled = random.sample(nodes, 2)
        prompt, processed_image = self.mark_and_prompt(
            sampled, image, self.absolute_distance_prompt_func, mark_prob=1.0
        )
        cog_ctx = self._make_singleview_cog_context(graph, sampled)
        return prompt, processed_image, QuestionType.OPEN_ENDED, cog_ctx

    def _generate_relative_distance(self, graph):
        """Sample three objects and ask relative distance to the third.

        Generates both an OE and an MCQ prompt from the same triple.
        Returns list[(prompt, image, qtype, cog_ctx)] with correct types per prompt.
        """
        nodes = graph.get_object_nodes()
        if len(graph.obj_tags) <= 2:
            return None
        image = graph.primary_view.image
        sampled = random.sample(nodes, 3)

        # mark all three objects at once
        processed_image, marked = self.marker.mark_objects(image, sampled)
        A, B, C = marked
        cog_ctx = self._make_singleview_cog_context(graph, sampled)

        return [
            (self.relative_distance_oe_prompt_func(A, B, C), processed_image, QuestionType.OPEN_ENDED, cog_ctx),
            (self.relative_distance_mcq_prompt_func(A, B, C), processed_image, QuestionType.MCQ, cog_ctx),
        ]