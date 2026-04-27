import random
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType

from utils.point_cloud_utils import compute_point_cloud_distance


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Distance"
    SUB_TASKS = {
        "pair_absolute_distance":  {"default": 1, "handler": "_generate_pair_absolute_distance"},
        "multi_relative_distance": {"default": 1, "handler": "_generate_multi_relative_distance"},
    }

    def get_mark_config(self):
        return MarkConfig(type_weights={"mask": 0.3, "box": 0.7})

    # ─── Prompt Functions ─────────────────────────────────────────────

    def pair_absolute_distance_prompt_func(self, A, B):
        """Generate an absolute distance QA for two objects from different views."""
        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        A_cloud = self._clean_cloud(A_cloud)
        B_cloud = self._clean_cloud(B_cloud)

        unit = random.choice(["m", "cm"])
        dist = compute_point_cloud_distance(A_cloud, B_cloud)
        scaled = dist * self.scaling_factor * (100 if unit == "cm" else 1)

        return self.render_prompt(
            f"distance.absolute_{unit}",
            shared={"A": A_desc, "B": B_desc, "X": f"{scaled:.2f} {unit}"},
        )

    def multi_relative_distance_prompt_func(self, obj_infos):
        """Generate a relative distance QA for N objects from different views."""
        distance_type = random.choice(["closest", "farthest"])

        ref_idx = random.randint(0, len(obj_infos) - 1)
        ref_desc, ref_cloud = obj_infos[ref_idx]
        ref_desc = ref_desc.lower()
        ref_cloud = self._clean_cloud(ref_cloud)

        candidates = [obj_infos[i] for i in range(len(obj_infos)) if i != ref_idx]
        distances = []
        for cand_desc, cand_cloud in candidates:
            cand_cloud = self._clean_cloud(cand_cloud)
            dist = compute_point_cloud_distance(ref_cloud, cand_cloud)
            distances.append((dist, cand_desc.lower()))

        distances.sort(key=lambda x: x[0], reverse=(distance_type == "farthest"))
        target_desc = distances[0][1]
        all_descs = [d for _, d in distances]
        random.shuffle(all_descs)

        return self.render_prompt(
            f"distance.{distance_type}",
            shared={"T": ", ".join(all_descs)},
            q_args={"X": ref_desc},
            a_args={"X": target_desc, "Y": ref_desc},
        )

    # ─── Handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_pair_absolute_distance(self, graph):
        if not graph.is_metric_depth:
            return None
        result = self._find_chain_and_mark(graph, num_views=2)
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt = self.pair_absolute_distance_prompt_func(objs[0], objs[1])
        cog_ctx = self._collect_cog_context_from_meta(meta)
        return prompt, processed_images, QuestionType.OPEN_ENDED, cog_ctx

    def _generate_multi_relative_distance(self, graph):
        result = self._find_chain_and_mark(graph, num_views=random.choice([3, 4, 5, 6]))
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt = self.multi_relative_distance_prompt_func(objs)
        cog_ctx = self._collect_cog_context_from_meta(meta)
        return prompt, processed_images, QuestionType.OPEN_ENDED, cog_ctx
