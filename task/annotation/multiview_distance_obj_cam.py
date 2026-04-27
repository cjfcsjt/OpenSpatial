import random
import numpy as np
import open3d as o3d
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.visual_marker import MarkConfig
from .core.question_type import QuestionType


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Distance"
    SUB_TASKS = {
        "object_camera_distance": {"default": 1, "handler": "_generate_object_camera_distance"},
    }

    def get_mark_config(self):
        return MarkConfig(mark_types=["mask", "box"])

    def __init__(self, args):
        super().__init__(args)
        self.dis_pot_thre = args.get("dis_pot_thre", 0.08)

    # ─── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _mean_closest_distance_to_origin(cloud, ratio=0.2):
        """Mean distance of the closest `ratio` points in `cloud` to the origin."""
        origin = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]])))
        dists = cloud.compute_point_cloud_distance(origin)
        k = max(1, int(len(dists) * ratio))
        return float(np.mean(np.partition(dists, k)[:k]))

    # ─── Prompt Function ─────────────────────────────────────────────

    def obj_cam_distance_prompt_func(self, A, B):
        """Generate an MCQ asking in which view an object is closer/farther to the camera.

        Computes mean camera-frame distance for the object in each view,
        then builds a 3-option MCQ (View 1 / View 2 / equal).
        """
        A_desc, A_cloud = A
        B_desc, B_cloud = B
        A_desc = A_desc.lower()

        A_cloud = self._clean_cloud(A_cloud)
        B_cloud = self._clean_cloud(B_cloud)

        close_far = random.choice(["closer", "farther"])
        dist_A = self._mean_closest_distance_to_origin(A_cloud)
        dist_B = self._mean_closest_distance_to_origin(B_cloud)

        # Determine answer
        if abs(dist_A - dist_B) < self.dis_pot_thre:
            answer = "C"
        elif (dist_A < dist_B) == (close_far == "closer"):
            answer = "A"
        else:
            answer = "B"

        # Build option text from answer templates
        tpl = self.get_template("distance.obj_cam")
        _, opt_tpl = tpl.sample()
        opt_a = tpl._fill(opt_tpl, {"A": A_desc, "Y": close_far, "X": "View 1"})
        opt_b = tpl._fill(opt_tpl, {"A": A_desc, "Y": close_far, "X": "View 2"})
        opt_c = "distance to the spot where camera View 1 and View 2 were positioned is equal"
        options_str = f"Options: A. {opt_a}\nB. {opt_b}\nC. {opt_c}"

        return self.render_prompt(
            "distance.obj_cam_mcq",
            shared={"A": A_desc, "Y": close_far, "O": options_str, "X": answer},
        )

    # ─── Data Finder ─────────────────────────────────────────────────

    def _find_shared_obj_in_views(self, graph):
        """Find the same object visible in two different views (camera-frame pointclouds).

        Uses _find_overlapping_views to locate an anchor node in two diverse views,
        then collects camera-frame pointclouds for object-to-camera distance comparison.

        Returns:
            (meta_data, True) or (None, False).
            meta_data keys: image, mask, tag, view_idx, pointcloud, bbox_2d, node_ids.
        """
        node, views = self._find_overlapping_views(graph, num_views=2)
        if node is None:
            return None, False

        v1, v2 = views
        app1 = node.view_appearances[v1]
        app2 = node.view_appearances[v2]

        # Guard against missing pointcloud data
        if app1.pointcloud_camera is None or app2.pointcloud_camera is None:
            return None, False

        meta = {
            "image": [graph.views[v1].image, graph.views[v2].image],
            "mask": [app1.mask, app2.mask],
            "tag": [node.tag, node.tag],
            "view_idx": [v1, v2],
            "pointcloud": [app1.pointcloud_camera, app2.pointcloud_camera],
            "bbox_2d": [app1.bbox_2d, app2.bbox_2d],
            "node_ids": [node.node_id],
        }
        return meta, True

    # ─── Handler (dispatched by SUB_TASKS) ────────────────────────────

    def _generate_object_camera_distance(self, graph):
        if not graph.is_metric_depth:
            return None

        for _ in range(5):
            meta, flag = self._find_shared_obj_in_views(graph)
            if flag:
                break
        if not flag:
            return None

        mark_type = self.marker.choose_mark_type()
        processed_images, objs = self._mark_per_view(meta, mark_type)
        prompt = self.obj_cam_distance_prompt_func(objs[0], objs[1])
        cog_ctx = self._collect_cog_context_from_meta(meta)
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
