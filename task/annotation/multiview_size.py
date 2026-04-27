import random
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Size"
    SUB_TASKS = {
        "pair_relative_size":  {"default": 1, "handler": "_generate_pair_relative_size"},
        "multi_relative_size": {"default": 1, "handler": "_generate_multi_relative_size"},
    }

    def get_mark_config(self):
        return MarkConfig(mark_types=["mask", "box"])

    # ─── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_volume(cloud, box_3d_world=None):
        """Compute volume from 3D box (preferred) or AABB fallback."""
        if box_3d_world is not None:
            ext = box_3d_world[3:6]
        else:
            ext = cloud.get_axis_aligned_bounding_box().get_extent()
        return ext[0] * ext[1] * ext[2]

    # ─── Prompt Functions ─────────────────────────────────────────────

    def pair_relative_size_prompt_func(self, A, B, boxes_3d_world=None):
        """Generate a relative size QA for two objects from different views."""
        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc, B_desc = A_desc.lower(), B_desc.lower()

        volume_A = self._get_volume(A_cloud, boxes_3d_world[0] if boxes_3d_world else None)
        volume_B = self._get_volume(B_cloud, boxes_3d_world[1] if boxes_3d_world else None)

        tpl_name = random.choice(["size.big.multi_view", "size.small.multi_view"])
        is_bigger = volume_A > volume_B
        condition = is_bigger if "big" in tpl_name else not is_bigger

        return self.render_prompt(tpl_name, condition=condition, shared={"A": A_desc, "B": B_desc})

    def multi_relative_size_prompt_func(self, obj_infos, boxes_3d_world=None):
        """Generate a superlative size QA for N objects from different views."""
        size_type = random.choice(["biggest", "smallest"])
        tpl_name = f"size.{size_type}"

        volumes = []
        for i, (desc, cloud) in enumerate(obj_infos):
            vol = self._get_volume(cloud, boxes_3d_world[i] if boxes_3d_world else None)
            volumes.append((vol, desc.lower()))

        volumes.sort(key=lambda x: x[0], reverse=(size_type == "biggest"))
        target_desc = volumes[0][1]
        all_tags = [d for _, d in volumes]
        random.shuffle(all_tags)

        return self.render_prompt(
            tpl_name, shared={"T": ", ".join(all_tags), "X": target_desc},
        )

    # ─── Handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_pair_relative_size(self, graph):
        result = self._find_chain_and_mark(graph, num_views=2)
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt = self.pair_relative_size_prompt_func(objs[0], objs[1], boxes_3d_world=meta["box_3d_world"])
        cog_ctx = self._collect_cog_context_from_meta(meta)
        return prompt, processed_images, QuestionType.OPEN_ENDED, cog_ctx

    def _generate_multi_relative_size(self, graph):
        result = self._find_chain_and_mark(graph, num_views=random.choice([3, 4, 5, 6]))
        if result is None:
            return None
        meta, processed_images, objs = result
        prompt = self.multi_relative_size_prompt_func(objs, boxes_3d_world=meta["box_3d_world"])
        cog_ctx = self._collect_cog_context_from_meta(meta)
        return prompt, processed_images, QuestionType.OPEN_ENDED, cog_ctx
