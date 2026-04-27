"""
BEV (bird's-eye-view) pose estimation task — all-angles style.

Picks 3 pose-diverse views, then renders the BEV layout plus 3 perturbed
distractors. The MCQ asks which BEV diagram correctly depicts the 3 cameras'
spatial layout.

Output images: 3 RGB views + 4 BEV option diagrams (A/B/C/D).
Answer: letter of the true BEV diagram.
"""

import random

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from .core.cognitive_map import (
    CognitiveMapBuilder,
    CognitiveMapContext,
    CognitiveMapRenderer,
    generate_bev_perturbations,
)
from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "BEV Pose Estimation"
    SUB_TASKS = {
        "bev_pose_mcq": {"default": 1, "handler": "_generate_bev_pose_mcq"},
    }

    def __init__(self, args):
        super().__init__(args)
        # This task is always a BEV-rendering task; it always needs a builder
        # and renderer regardless of whether the global cognitive_map feature
        # is enabled, so we keep a private instance.
        self._bev_builder = CognitiveMapBuilder(grid_size=10, padding_ratio=0.15)
        self._bev_renderer = CognitiveMapRenderer(figsize=(5.0, 5.0), dpi=110)

    # ─── Handler ─────────────────────────────────────────────────────

    def _pick_three_views(self, graph, retries=10):
        view_ids = [vi for vi in graph.views if graph.views[vi].pose is not None]
        if len(view_ids) < 3:
            return None
        for _ in range(retries):
            triple = random.sample(view_ids, 3)
            poses = [graph.views[v].pose for v in triple]
            # Ensure some pose diversity between all three.
            ok = True
            for i in range(3):
                others = [poses[j] for j in range(3) if j != i]
                if not self._check_pose_diversity(poses[i], others,
                                                  self.min_rot_angle,
                                                  self.min_translation):
                    ok = False
                    break
            if ok:
                return triple
        return None

    def _generate_bev_pose_mcq(self, graph):
        triple = self._pick_three_views(graph)
        if triple is None:
            return None

        ctx = CognitiveMapContext(view_indices=list(triple))
        true_map = self._bev_builder.build(graph, ctx)
        if true_map is None:
            return None

        # 3 perturbations to act as distractors.
        rng = random.Random()
        distractors = generate_bev_perturbations(true_map, n=3, rng=rng)
        if len(distractors) < 3:
            return None

        all_maps = [true_map] + distractors
        shuffled, answer_letter = self._shuffle_mcq(all_maps, correct_idx=0)

        bev_images = []
        for i, cmap in enumerate(shuffled):
            png = self._bev_renderer.render_bev_only(
                cmap, title=f"Option {'ABCD'[i]}"
            )
            if png is None:
                return None
            bev_images.append({"bytes": png})

        rgb_images = [{"bytes": convert_pil_to_bytes(graph.views[v].image)}
                      for v in triple]

        options_str = "Options: A / B / C / D (see diagrams)"
        question = (
            "Below are 3 RGB views (views 1, 2, 3) followed by 4 bird's-eye-view "
            "diagrams labeled A-D. Which BEV diagram correctly represents the "
            "spatial layout of the three cameras? " + options_str
        )
        prompt = question + " Answer: " + answer_letter

        cog_ctx = self._make_cog_context(view_indices=list(triple))
        return prompt, rgb_images + bev_images, QuestionType.MCQ, cog_ctx
