"""
Multiview point correspondence annotation task.

Given two views of the same scene, finds a 3D point visible in both views,
marks a query point on View 1, and asks the model to identify the
corresponding point among labeled candidates on View 2.

Sub-tasks:
    point_correspondence_oe  — open-ended: model outputs a label (e.g. "point-B")
    point_correspondence_mcq — multiple choice: model picks from A/B/C/D options

Algorithm (vectorized Open3D):
    1. Use _find_overlapping_views to find two diverse views sharing a common object.
    2. Backproject valid depth pixels to 3D world coordinates for both views.
    3. Use Open3D's compute_point_cloud_distance for vectorized nearest-neighbor
       overlap detection within `overlap_dist` threshold.
    4. Randomly select one overlapping point, project it back to 2D in both views.
    5. Generate 3 distractor points on View 2 (at least `min_distractor_dist`
       pixels away from the ground-truth point).
    6. Shuffle GT + distractors, assign labels (A/B/C/D or 1/2/3/4).

Templates used:
    correspondence.point2point     — [A] query color, [B] candidate color, [T] answer label
    correspondence.point2point_num — same, but with numeric labels (1/2/3/4)

Configurable parameters (via YAML args):
    overlap_dist          — 3D distance threshold (meters) for two points to be
                            considered overlapping (default: 0.003)
    min_overlap_points    — minimum number of overlapping 3D points required to
                            accept a view pair (default: 10)
    boundary_margin       — pixel margin from image edge; points too close to
                            the border are rejected (default: 10)
    min_distractor_dist   — minimum pixel distance between each distractor and
                            the ground-truth point on View 2 (default: 20)
"""

import random
import numpy as np
import open3d as o3d
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Correspondence"
    SUB_TASKS = {
        "point_correspondence_oe":  {"default": 1, "handler": "_generate_point_correspondence_oe"},
        "point_correspondence_mcq": {"default": 1, "handler": "_generate_point_correspondence_mcq"},
    }

    def __init__(self, args):
        super().__init__(args)
        self.overlap_dist = args.get("overlap_dist", 0.003)
        self.min_overlap_points = args.get("min_overlap_points", 10)
        self.boundary_margin = args.get("boundary_margin", 10)
        self.min_distractor_dist = args.get("min_distractor_dist", 20)

    # ─── Prompt Function ──────────────────────────────────────────────

    def point_correspondence_prompt_func(self, question_color, candidate_color, gt_answer, question_type=QuestionType.OPEN_ENDED):
        """Build a point correspondence QA string.

        Args:
            question_color:  color name of the query point on View 1 (e.g. "red").
            candidate_color: color name of candidate points on View 2 (e.g. "blue").
            gt_answer:       ground-truth label — "A"/"B"/"C"/"D" or "1"/"2"/"3"/"4".
            question_type:   QuestionType.OPEN_ENDED uses render_prompt; QuestionType.MCQ uses render_qa
                             with appended options string.

        Returns:
            Formatted "question Answer: answer" string.
        """
        is_num = gt_answer in ["1", "2", "3", "4"]
        tpl_name = "correspondence.point2point_num" if is_num else "correspondence.point2point"

        if question_type == QuestionType.MCQ:
            # Build option markers and find the correct one
            if is_num:
                markers = ["A point-1", "B point-2", "C point-3", "D point-4"]
                idx = ["1", "2", "3", "4"].index(gt_answer)
            else:
                markers = ["A point-A", "B point-B", "C point-C", "D point-D"]
                idx = ["A", "B", "C", "D"].index(gt_answer)
            question, answer = self.get_template(tpl_name).render_qa(
                shared={"A": question_color, "B": candidate_color},
                a_args={"T": markers[idx]},
            )
            options = "\nOptions: " + " ".join(markers)
            return question + " " + options + " Answer: " + answer
        else:
            # Open-ended: 70% "point-X" format, 30% bare label (letter only)
            if is_num:
                target = "point-" + gt_answer
            else:
                target = ("point-" + gt_answer) if random.random() < 0.7 else gt_answer
            return self.render_prompt(
                tpl_name, shared={"A": question_color, "B": candidate_color, "T": target},
            )

    # ─── Point Correspondence Finder ──────────────────────────────────

    def _find_point_correspondence(self, graph):
        """Find a corresponding 3D point visible in two overlapping views.

        Returns:
            (meta_data, True) on success, where meta_data contains:
                image:      [PIL.Image, PIL.Image] — the two view images
                view_idx:   [int, int]             — view indices in the SceneGraph
                point:      [pt1_uv, pt2_uv, distractor_uvs]
                            pt1_uv:         [u, v] query point on View 1
                            pt2_uv:         [u, v] ground-truth point on View 2
                            distractor_uvs: [[u,v], ...] × 3 distractor points on View 2
            (None, False) on failure.
        """
        # Step 1: get two diverse views sharing a common object
        node, views = self._find_overlapping_views(graph, num_views=2)
        if node is None:
            return None, False

        view1_idx, view2_idx = views
        view1 = graph.views[view1_idx]
        view2 = graph.views[view2_idx]

        intrinsic = view1.intrinsic
        depth_map1 = view1.depth_map
        depth_map2 = view2.depth_map
        h, w = depth_map1.shape
        img_dim = (w, h)

        # Step 2: backproject valid depth pixels to 3D world coordinates
        points_3d_1 = self.backproject_2d_to_3d(view1.pose, depth_map1, img_dim, intrinsic)
        points_3d_2 = self.backproject_2d_to_3d(view2.pose, depth_map2, img_dim, intrinsic)

        valid1 = depth_map1.ravel() > 0
        valid2 = depth_map2.ravel() > 0
        pts1_valid = points_3d_1[valid1]
        pts2_valid = points_3d_2[valid2]

        min_pts = self.min_overlap_points
        if len(pts1_valid) < min_pts or len(pts2_valid) < min_pts:
            return None, False

        # Step 3: compute nearest-neighbor distances (vectorized via Open3D)
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()

        # Subsample View 1 queries if too large to keep runtime bounded
        max_query = 50000
        if len(pts1_valid) > max_query:
            query_idx = np.random.choice(len(pts1_valid), max_query, replace=False)
            pts1_query = pts1_valid[query_idx]
        else:
            pts1_query = pts1_valid

        pcd1.points = o3d.utility.Vector3dVector(pts1_query)
        pcd2.points = o3d.utility.Vector3dVector(pts2_valid)

        # compute_point_cloud_distance returns nearest-neighbor distance per point
        nn_dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        overlap_indices = np.where(nn_dists < self.overlap_dist)[0]

        if len(overlap_indices) < min_pts:
            return None, False

        # Step 4: pick a random overlapping point, project back to 2D in both views
        sel_local = random.choice(overlap_indices)
        sel_pt = pts1_query[sel_local]
        # Single-point nearest neighbor via numpy broadcast (faster than building KDTree)
        corr_idx = np.argmin(np.sum((pts2_valid - sel_pt) ** 2, axis=1))
        corr_pt = pts2_valid[corr_idx]

        inv_pose1 = np.linalg.inv(view1.pose)
        inv_pose2 = np.linalg.inv(view2.pose)
        pt1_uv = self.project_3d_to_2d(inv_pose1, sel_pt.reshape(1, 3), intrinsic)[0]
        pt2_uv = self.project_3d_to_2d(inv_pose2, corr_pt.reshape(1, 3), intrinsic)[0]

        # Reject if either projected point is too close to the image boundary
        margin = self.boundary_margin
        if not (margin <= pt1_uv[0] <= w - margin and margin <= pt1_uv[1] <= h - margin):
            return None, False
        if not (margin <= pt2_uv[0] <= w - margin and margin <= pt2_uv[1] <= h - margin):
            return None, False

        # Step 5: generate 3 distractor points on View 2 (far enough from GT)
        min_dist_sq = self.min_distractor_dist ** 2
        uv_candidates = []
        for _ in range(100):
            u = random.randint(margin, w - margin - 1)
            v = random.randint(margin, h - margin - 1)
            if (u - pt2_uv[0]) ** 2 + (v - pt2_uv[1]) ** 2 < min_dist_sq:
                continue
            uv_candidates.append([u, v])
            if len(uv_candidates) >= 3:
                break
        if len(uv_candidates) < 3:
            return None, False

        meta_data = {
            "image": [view1.image, view2.image],
            "view_idx": [view1_idx, view2_idx],
            "point": [pt1_uv.astype(int).tolist(), pt2_uv.astype(int).tolist(), uv_candidates],
        }
        return meta_data, True

    # ─── Visual Marking ──────────────────────────────────────────────

    def _draw_candidate_points(self, image1, image2, point1_uv, point2_uv, candidates_uv):
        """Draw query point on View 1 and labeled candidate points on View 2.

        View 1 gets a single unlabeled point (the query).
        View 2 gets 4 labeled points (1 GT + 3 distractors) in shuffled order.

        Args:
            image1, image2: PIL images for the two views.
            point1_uv:      [u, v] query point on View 1.
            point2_uv:      [u, v] ground-truth corresponding point on View 2.
            candidates_uv:  [[u, v], ...] × 3 distractor points on View 2.

        Returns:
            (processed_image1, processed_image2,
             color_name1, color_name2, gt_answer)
            where gt_answer is the label assigned to the GT point after shuffling.
        """
        # View 1: single query point (no label)
        processed_image1, color_name1 = self.marker.mark_objects(image1, points=[point1_uv])

        # Shuffle GT among distractors so the model can't exploit position
        all_points = [point2_uv] + candidates_uv
        indices = list(range(len(all_points)))
        random.shuffle(indices)
        labels = ["1", "2", "3", "4"] if random.random() < 0.3 else ["A", "B", "C", "D"]
        shuffled = [all_points[i] for i in indices]
        gt_answer = labels[indices.index(0)]  # label assigned to the GT point (index 0)

        # View 2: labeled candidate points
        processed_image2, color_name2 = self.marker.mark_objects(
            image2, points=shuffled, labels=labels,
        )

        return processed_image1, processed_image2, color_name1, color_name2, gt_answer

    # ─── Handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _build_correspondence(self, graph, question_type):
        """Shared pipeline for both OE and MCQ handlers.

        Retries up to 5 times to find a valid point correspondence,
        then draws visual marks and generates the prompt.
        """
        for _ in range(5):
            meta, flag = self._find_point_correspondence(graph)
            if flag:
                break
        if not flag:
            return None

        point1, point2, uv_candidates = meta["point"]
        image1, image2 = meta["image"]

        self.marker.reset(shuffle=True)
        img1, img2, color1, color2, gt_answer = self._draw_candidate_points(
            image1, image2, point1, point2, uv_candidates)

        prompt = self.point_correspondence_prompt_func(color1, color2, gt_answer, question_type)
        qtype = QuestionType.MCQ if question_type == QuestionType.MCQ else QuestionType.OPEN_ENDED

        cog_ctx = self._make_cog_context(view_indices=meta["view_idx"])
        return prompt, [img1, img2], qtype, cog_ctx

    def _generate_point_correspondence_oe(self, graph):
        """Generate an open-ended point correspondence QA."""
        return self._build_correspondence(graph, QuestionType.OPEN_ENDED)

    def _generate_point_correspondence_mcq(self, graph):
        """Generate an MCQ point correspondence QA."""
        return self._build_correspondence(graph, QuestionType.MCQ)
