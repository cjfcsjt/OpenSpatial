import math
import random
from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.visual_marker import MarkConfig
from utils.box_utils import check_box_3d_vertical_overlap
from .core.question_type import QuestionType

from utils.image_utils import convert_pil_to_bytes


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "Position"
    SUB_TASKS = {
        "position_oe":  {"default": 1, "handler": "_generate_position_oe"},
        "position_mcq": {"default": 1, "handler": "_generate_position_mcq"},
    }

    def get_mark_config(self):
        return MarkConfig(mark_types=["mask", "box"])

    _ANGLES = [0, 45, 90, 135, 180, -135, -90, -45]
    _DIR_TEMPLATES = {
        "type1": ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"],
        "type2": ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"],
    }
    _DIR_MAPS = {
        "type1": dict(zip(_DIR_TEMPLATES["type1"], _ANGLES)),
        "type2": dict(zip(_DIR_TEMPLATES["type2"], _ANGLES)),
    }

    # ─── Direction Logic ──────────────────────────────────────────────

    @staticmethod
    def get_direction(dx, dy, dir_tmp, delta=15):
        angle = math.degrees(math.atan2(dx, dy))
        if -delta < angle <= delta:
            return dir_tmp[0]
        elif delta < angle <= 90-delta:
            return dir_tmp[1]
        elif 90-delta < angle <= 90+delta:
            return dir_tmp[2]
        elif 90+delta < angle <= 180-delta:
            return dir_tmp[3]
        elif angle > 180-delta or angle <= -180+delta:
            return dir_tmp[4]
        elif -180+delta < angle <= -90-delta:
            return dir_tmp[5]
        elif -90-delta < angle <= -90+delta:
            return dir_tmp[6]
        elif -90+delta < angle <= -delta:
            return dir_tmp[7]
        else:
            return 'unknown'

    @staticmethod
    def rotate(dx, dy, dx1, dy1, prior_direction, dir_map):
        actual_angle = math.degrees(math.atan2(dx1, dy1))
        target_angle = dir_map.get(prior_direction, 0)
        rotate_angle = target_angle - actual_angle
        rad = math.radians(-rotate_angle)
        new_dx = dx * math.cos(rad) - dy * math.sin(rad)
        new_dy = dx * math.sin(rad) + dy * math.cos(rad)
        return new_dx, new_dy

    def relative_direction(self, p1, p2, p3, template_type):
        dir_tmp = self._DIR_TEMPLATES[template_type]
        dir_map = self._DIR_MAPS[template_type]
        prior_direction = dir_tmp[random.choice([0, 2, 4, 6])]
        dx1, dy1 = p1[0] - p2[0], p1[1] - p2[1]
        dx3, dy3 = p3[0] - p2[0], p3[1] - p2[1]
        new_dx3, new_dy3 = self.rotate(dx3, dy3, dx1, dy1, prior_direction, dir_map)
        level1_direction = self.get_direction(new_dx3, new_dy3, dir_tmp, delta=10)
        level2_direction = self.get_direction(new_dx3, new_dy3, dir_tmp, delta=15)
        level3_direction = self.get_direction(new_dx3, new_dy3, dir_tmp, delta=20)
        return prior_direction, [level1_direction, level2_direction, level3_direction]

    # ─── Prompt Functions ─────────────────────────────────────────────

    def _build_mcq_options(self, dir_B2anchor, dir_B2anchors):
        """Build MCQ options string and the answer token."""
        dir_tmp = self._DIR_TEMPLATES["type1"] if dir_B2anchor in self._DIR_TEMPLATES["type1"] else self._DIR_TEMPLATES["type2"]
        wrong_options = [d for d in dir_tmp if d != dir_B2anchor and d not in dir_B2anchors]
        wrong_options = random.sample(wrong_options, 3)
        candidates = [dir_B2anchor] + wrong_options
        shuffled, answer_letter = self._shuffle_mcq(candidates)

        options_list = [f"{'ABCD'[i]}.{shuffled[i]}" for i in range(4)]
        options_str = "Options: " + " ".join(options_list) + "."
        # 50% full answer "A.north", 50% letter only "A"
        if random.random() < 0.5:
            answer_option = f"{answer_letter}.{dir_B2anchor}"
        else:
            answer_option = answer_letter
        return options_str, answer_option

    def position_prompt_func(self, A, B, anchor, question_type=QuestionType.OPEN_ENDED):
        """Generate a position QA (open-ended or MCQ).

        Args:
            A: (desc, box_3d_world) for object in View 1.
            B: (desc, box_3d_world) for object in View 2.
            anchor: (desc, box_3d_world) for the anchor object visible in both views.
            question_type: QuestionType.OPEN_ENDED or QuestionType.MCQ.

        Returns:
            Formatted "question Answer: answer" string, or None if direction is unknown.
        """
        A_desc, A_box_3d_world = A
        B_desc, B_box_3d_world = B
        anchor_desc, anchor_box_3d_world = anchor

        template_type = random.choice(["type1", "type2"])
        dir_A2anchor, dir_B2anchors = self.relative_direction(
            A_box_3d_world[:2], anchor_box_3d_world[:2], B_box_3d_world[:2], template_type)
        if dir_A2anchor == 'unknown' or 'unknown' in dir_B2anchors:
            return None

        dir_B2anchor = dir_B2anchors[random.choice([1, 2])]
        shared = {"A": A_desc, "B": anchor_desc, "C": B_desc, "X": dir_A2anchor}

        if question_type == QuestionType.OPEN_ENDED:
            return self.render_prompt(
                f"multiview_position.{template_type}",
                shared={**shared, "T": dir_B2anchor},
            )
        else:
            options_str, answer_option = self._build_mcq_options(dir_B2anchor, dir_B2anchors)
            return self.render_prompt(
                f"multiview_position.{template_type}_mcq",
                shared={**shared, "T": answer_option, "O": options_str},
            )

    # ─── Data Finder ──────────────────────────────────────────────────

    def _find_pair_from_overlapping_views(self, graph):
        """Find anchor (in both views) + 2 unique objects (one per view).

        Uses _find_overlapping_views to locate an anchor node in two views,
        then finds objects unique to each view for position reasoning.

        Returns:
            (meta_data, True) or (None, False).
        """
        anchor_node, views = self._find_overlapping_views(graph, num_views=2, pose_diversity=False)
        if anchor_node is None:
            return None, False
        view1_idx, view2_idx = views

        # Find tags unique to each view (single traversal)
        view_data = self._tags_and_nodes_in_views(graph, [view1_idx, view2_idx])
        tags_v1 = set(view_data[view1_idx].keys())
        tags_v2 = set(view_data[view2_idx].keys())
        only_in_v1 = tags_v1 - tags_v2
        only_in_v2 = tags_v2 - tags_v1
        if not only_in_v1 or not only_in_v2:
            return None, False

        obj1_tag = random.choice(list(only_in_v1))
        obj2_tag = random.choice(list(only_in_v2))

        node1 = view_data[view1_idx][obj1_tag]
        node2 = view_data[view2_idx][obj2_tag]

        # Validate all 3D boxes exist and have no vertical overlap
        anchor_box_3d_world = anchor_node.box_3d_world
        if anchor_box_3d_world is None or node1.box_3d_world is None or node2.box_3d_world is None:
            return None, False
        if check_box_3d_vertical_overlap([anchor_box_3d_world, node1.box_3d_world, node2.box_3d_world]):
            return None, False

        app1 = node1.view_appearances[view1_idx]
        app2 = node2.view_appearances[view2_idx]
        anchor_app1 = anchor_node.view_appearances[view1_idx]

        meta = {
            "image": [graph.views[view1_idx].image, graph.views[view2_idx].image],
            "mask": [app1.mask, app2.mask],
            "tag": [obj1_tag, obj2_tag],
            "view_idx": [view1_idx, view2_idx],
            "bbox_2d": [app1.bbox_2d, app2.bbox_2d],
            "box_3d_world": [node1.box_3d_world, node2.box_3d_world],
            "anchor_tag": anchor_node.tag,
            "anchor_box_3d_world": anchor_box_3d_world,
            "anchor_bbox_2d": anchor_app1.bbox_2d,
            "anchor_mask": anchor_app1.mask,
        }
        return meta, True

    # ─── Marking + Prompt ─────────────────────────────────────────────

    def _build_pair_position_prompt(self, graph, question_type):
        for _ in range(20):
            meta, flag = self._find_pair_from_overlapping_views(graph)
            if flag:
                break
        if not flag:
            return None

        mark_type = self.marker.choose_mark_type()
        if random.random() < 0.5:
            # Mark objects on images
            A_obj = (meta["tag"][0], meta["box_3d_world"][0],
                     meta["bbox_2d"][0], meta["mask"][0])
            anchor_obj = (meta["anchor_tag"], meta["anchor_box_3d_world"],
                          meta["anchor_bbox_2d"], meta["anchor_mask"])
            B_obj = (meta["tag"][1], meta["box_3d_world"][1],
                     meta["bbox_2d"][1], meta["mask"][1])

            self.marker.reset(shuffle=True)
            processed_image1, info1 = self.marker.mark_objects(
                meta["image"][0], [A_obj, anchor_obj], mark_type=mark_type)
            A_info, anchor_info = info1[0], info1[1]
            processed_image2, info2 = self.marker.mark_objects(
                meta["image"][1], [B_obj], mark_type=mark_type)
            B_info = info2[0]
        else:
            # Plain images, no marking
            processed_image1 = {"bytes": convert_pil_to_bytes(meta["image"][0])}
            processed_image2 = {"bytes": convert_pil_to_bytes(meta["image"][1])}
            A_info = (meta["tag"][0], meta["box_3d_world"][0])
            B_info = (meta["tag"][1], meta["box_3d_world"][1])
            anchor_info = (meta["anchor_tag"], meta["anchor_box_3d_world"])

        prompt = self.position_prompt_func(A_info, B_info, anchor_info, question_type)
        if prompt is None:
            return None

        cog_ctx = self._collect_cog_context_from_meta(meta)
        return prompt, [processed_image1, processed_image2], cog_ctx

    # ─── Handlers (dispatched by SUB_TASKS) ───────────────────────────

    def _generate_position_oe(self, graph):
        result = self._build_pair_position_prompt(graph, question_type=QuestionType.OPEN_ENDED)
        if result is None:
            return None
        prompt, processed_images, cog_ctx = result
        return prompt, processed_images, QuestionType.OPEN_ENDED, cog_ctx

    def _generate_position_mcq(self, graph):
        result = self._build_pair_position_prompt(graph, question_type=QuestionType.MCQ)
        if result is None:
            return None
        prompt, processed_images, cog_ctx = result
        return prompt, processed_images, QuestionType.MCQ, cog_ctx
