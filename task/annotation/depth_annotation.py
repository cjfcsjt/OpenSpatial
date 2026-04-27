"""
Depth estimation annotation task: depth ordering & depth choice.

Sub-tasks:
    depth_ordering_oe  — sort 3-5 marked objects/points by depth (near→far), open-ended.
    depth_ordering_mcq — same ordering task but as 4-option MCQ (permutation choices).
    depth_choice_oe    — pick the closest / farthest / N-th closest object, open-ended.
    depth_choice_mcq   — same choice task but as 4-option MCQ.

Visual annotation modes:
    random_sample (~10%) — sample 4-7 random pixels with distinct depths; returns
                           normalized [x, y] coordinate tags. No drawing on image.
    object-based  (~90%) — select 3-5 nodes, draw point/mask/box via VisualMarker
                           (same pattern as size / distance / position tasks),
                           then compute per-object depth from the depth_map.

Depth estimation:
    For each object, depth = mean of the shallowest 10% of valid mask pixels.
    This approximates the front-surface depth and is robust to noisy backgrounds.

Templates used:
    depth.ordering       — [T] type label, [A] object list, [X] sorted list
    depth.ordering_mcq   — [T] type label, [Y] options; [X] obj list (q) / answer (a)
    depth.farthest       — [T] type label, [A] object list, [X] answer
    depth.closest        — [T] type label, [A] object list, [X] answer
    depth.choice         — [T] type label, [A] object list, [B] ordinal, [X] answer
    depth.farthest_mcq   — [T] type label, [Y] options; [X] obj list (q) / answer (a)
    depth.closest_mcq    — [T] type label, [Y] options; [X] obj list (q) / answer (a)
    depth.choice_mcq     — [T] type label, [Y] ordinal, [Z] options; [X] obj list (q) / answer (a)
"""

import random
import numpy as np
from .core.base_annotation_task import BaseAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType

from utils.image_utils import convert_pil_to_bytes

ORDINALS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth",
]


class AnnotationGenerator(BaseAnnotationTask):

    QUESTION_TAG = "Depth Estimation"
    SUB_TASKS = {
        "depth_ordering_oe":  {"default": 1, "handler": "_generate_depth_ordering_oe"},
        "depth_ordering_mcq": {"default": 1, "handler": "_generate_depth_ordering_mcq"},
        "depth_choice_oe":    {"default": 1, "handler": "_generate_depth_choice_oe"},
        "depth_choice_mcq":   {"default": 1, "handler": "_generate_depth_choice_mcq"},
    }

    # ── config ────────────────────────────────────────────────────────

    def get_mark_config(self):
        """50% point, 25% mask, 25% box — point is favoured because depth
        questions focus on spatial position rather than object shape."""
        return MarkConfig(
            type_weights={"point": 0.5, "mask": 0.25, "box": 0.25},
        )

    def check_example(self, example) -> bool:
        """Require image, obj_tags, depth_map, masks, and >= 3 objects."""
        if not super().check_example(example):
            return False
        if "depth_map" not in example:
            return False
        if "masks" not in example:
            return False
        return len(example["obj_tags"]) >= 3

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_depth(mask, depth_map):
        """Estimate front-surface depth from a binary mask.

        Takes the shallowest 10% of valid (>0) depth pixels inside the mask
        and returns their mean. Falls back to the full mean when the mask
        covers very few pixels (k >= total count).
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        depths = depth_map[ys, xs]
        depths = depths[depths > 0]
        if len(depths) == 0:
            return None
        k = max(1, int(len(depths) * 0.1))
        if k >= len(depths):
            return float(np.mean(depths))
        # O(n) partial sort instead of O(n log n) full sort
        return float(np.mean(np.partition(depths, k)[:k]))


    # ── data preparation ─────────────────────────────────────────────

    def _sample_random_points(self, image, depth_map):
        """Generate coordinate-based tags by sampling random pixels.

        Picks 4-7 pixels whose depths differ by > 0.05 from all previously
        selected pixels. Coordinates are normalized to a [0, 1000] grid.
        The original image is returned unchanged (no drawing).

        Returns (tags, depth_sorted_tags, image_bytes) or None on failure.
        """
        h, w = depth_map.shape
        num_points = random.randint(4, 7)
        points, selected_depths = [], []
        for _ in range(num_points):
            for _ in range(100):
                u, v = random.randint(0, w - 1), random.randint(0, h - 1)
                d = depth_map[v, u]
                if d > 0 and all(abs(d - sd) > 0.05 for sd in selected_depths):
                    points.append([u, v])
                    selected_depths.append(d)
                    break
        if len(points) != num_points:
            return None
        sorted_pts = [points[i] for i in np.argsort(selected_depths)]
        norm = lambda p: str([int(p[0] / w * 1000), int(p[1] / h * 1000)])
        return (
            [norm(p) for p in points],
            [norm(p) for p in sorted_pts],
            {"bytes": convert_pil_to_bytes(image)},
        )

    def _mark_and_sort(self, image, depth_map, nodes):
        """Draw visual marks on 3-5 objects and sort them by depth.

        Delegates drawing to self.marker (VisualMarker), then computes
        per-object depth from the depth_map. Objects whose mask yields
        no valid depth are silently dropped.

        Returns (tags_with_color, depth_sorted_tags, image_bytes) or None.
        """
        num = random.randint(3, min(5, len(nodes)))
        sampled = random.sample(nodes, num)

        self.marker.reset(shuffle=True)
        processed_image, marked = self.marker.mark_objects(image, sampled)

        tags, depths = [], []
        for desc, node in marked:
            app = node.view_appearances.get(0)
            if app is None:
                continue
            mask = np.array(app.mask)
            depth = self._compute_depth(mask, depth_map)
            if depth is None:
                continue
            tags.append(desc)
            depths.append(depth)

        if len(tags) < 2:
            return None

        sorted_tags = [tags[i] for i in np.argsort(depths)]
        return (tags, sorted_tags, processed_image)

    def _prepare_marked_data(self, image, depth_map, nodes, qtype):
        """Top-level entry: choose annotation mode, produce depth-sorted tags.

        ~10% chance of random pixel sampling; otherwise object-based marking.
        If random sampling fails (too few distinct-depth pixels), falls
        through to object-based marking automatically.
        MCQ format requires >= 4 tags; returns None if insufficient.

        Returns (tags, depth_sorted_tags, image_bytes, t_label) or None.
        t_label is "points:" for random samples, "objects:" for node-based.
        """
        if random.random() < 0.1:
            result = self._sample_random_points(image, depth_map)
            if result is not None:
                tags, sorted_tags, image_bytes = result
                if qtype == QuestionType.MCQ and len(tags) < 4:
                    return None
                return tags, sorted_tags, image_bytes, "points:"

        result = self._mark_and_sort(image, depth_map, nodes)
        if result is None:
            return None
        tags, sorted_tags, image_bytes = result
        if qtype == QuestionType.MCQ and len(tags) < 4:
            return None
        return tags, sorted_tags, image_bytes, "objects:"

    def _prepare(self, graph):
        """Extract depth_map, image, and mask-bearing nodes from the graph.

        Returns (depth_map, image, filtered_nodes) or None when no nodes
        have masks in the primary view.
        """
        view = graph.primary_view
        depth_map = view.depth_map
        image = view.image

        assert image.size == depth_map.shape[::-1], \
            f"Image {image.size} vs depth_map {depth_map.shape[::-1]} dimension mismatch."

        nodes = [n for n in graph.get_object_nodes()
                 if n.view_appearances.get(0) and n.view_appearances[0].mask_path]
        if not nodes:
            return None
        return depth_map, image, nodes

    # ── prompt generation ────────────────────────────────────────────

    def _build_ordering_prompt(self, depth_map, image, nodes, qtype):
        """Build a depth-ordering QA pair.

        OE format:  question lists objects → answer is the near-to-far ordering.
        MCQ format: question lists objects + 4 permutation options (A-D) →
                    answer is the correct option letter.

        Returns (prompt_str, image_bytes) or (None, None).
        """
        marked = self._prepare_marked_data(image, depth_map, nodes, qtype)
        if marked is None:
            return None, None
        tags, sorted_tags, image_bytes, t_label = marked

        if qtype == QuestionType.OPEN_ENDED:
            prompt = self.render_prompt(
                "depth.ordering",
                shared={"T": t_label, "A": ', '.join(tags), "X": ', '.join(sorted_tags)},
            )
        else:
            # generate 3 distinct wrong orderings by random shuffling
            wrong_perms = []
            for _ in range(50):
                perm = sorted_tags[:]
                random.shuffle(perm)
                if perm != sorted_tags and perm not in wrong_perms:
                    wrong_perms.append(perm)
                    if len(wrong_perms) == 3:
                        break
            if len(wrong_perms) < 3:
                return None, None

            candidates = [sorted_tags] + wrong_perms
            shuffled, answer_option = self._shuffle_mcq(candidates)
            options = [f"{'ABCD'[i]}:{str(list(shuffled[i]))}" for i in range(4)]

            prompt = self.render_prompt(
                "depth.ordering_mcq",
                shared={"T": t_label, "Y": '\n'.join(options)},
                q_args={"X": ', '.join(tags)},
                a_args={"X": answer_option},
            )
        return prompt, image_bytes

    def _build_choice_prompt(self, depth_map, image, nodes, qtype):
        """Build a depth-choice QA pair.

        Randomly selects one of three question types:
            farthest (40%) — which object is farthest from the camera?
            closest  (40%) — which object is closest to the camera?
            choice   (20%) — which object is the N-th closest?

        Returns (prompt_str, image_bytes) or (None, None).
        """
        r = random.random()
        question_type = "farthest" if r < 0.4 else ("closest" if r < 0.8 else "choice")
        tpl_name = f"depth.{question_type}" + ("_mcq" if qtype == QuestionType.MCQ else "")

        marked = self._prepare_marked_data(image, depth_map, nodes, qtype)
        if marked is None:
            return None, None
        tags, sorted_tags, image_bytes, t_label = marked
        obj_str = ', '.join(tags)

        # correct answer position in the depth-sorted list
        if question_type == "farthest":
            correct_idx = len(sorted_tags) - 1
        elif question_type == "closest":
            correct_idx = 0
        else:
            correct_idx = random.randint(0, len(sorted_tags) - 1)

        if qtype == QuestionType.OPEN_ENDED:
            shared = {"T": t_label, "A": obj_str, "X": str(sorted_tags[correct_idx])}
            if question_type == "choice":
                shared["B"] = ORDINALS[correct_idx]
            prompt = self.render_prompt(tpl_name, shared=shared)
        else:
            # build A-D options: correct answer + 3 random wrong answers
            wrong_idx = [i for i in range(len(sorted_tags)) if i != correct_idx]
            candidates = [sorted_tags[correct_idx]] + [sorted_tags[i] for i in random.sample(wrong_idx, 3)]
            shuffled, answer_option = self._shuffle_mcq(candidates)
            options = [f"{'ABCD'[i]}:{str(shuffled[i])}" for i in range(4)]

            shared = {"T": t_label}
            if question_type == "choice":
                shared["Y"] = ORDINALS[correct_idx]
                shared["Z"] = '\n'.join(options)
            else:
                shared["Y"] = '\n'.join(options)

            prompt = self.render_prompt(
                tpl_name, shared=shared,
                q_args={"X": obj_str}, a_args={"X": answer_option},
            )
        return prompt, image_bytes

    # ── handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _dispatch(self, graph, task_kind, qtype):
        """Shared handler logic: prepare graph → build prompt → return result.

        Returns (prompt, image_bytes, QuestionType, cog_ctx) or None.
        """
        prepared = self._prepare(graph)
        if prepared is None:
            return None
        depth_map, image, nodes = prepared
        builder = self._build_ordering_prompt if task_kind == "ordering" else self._build_choice_prompt
        prompt, image_bytes = builder(depth_map, image, nodes, qtype)
        if prompt is None:
            return None
        cog_ctx = self._make_singleview_cog_context(graph, nodes)
        return prompt, image_bytes, qtype, cog_ctx

    def _generate_depth_ordering_oe(self, graph):
        return self._dispatch(graph, "ordering", QuestionType.OPEN_ENDED)

    def _generate_depth_ordering_mcq(self, graph):
        return self._dispatch(graph, "ordering", QuestionType.MCQ)

    def _generate_depth_choice_oe(self, graph):
        return self._dispatch(graph, "choice", QuestionType.OPEN_ENDED)

    def _generate_depth_choice_mcq(self, graph):
        return self._dispatch(graph, "choice", QuestionType.MCQ)
