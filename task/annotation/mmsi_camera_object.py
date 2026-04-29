"""
MMSI-Bench: Camera–Object direction task.

Two sub-tasks are supported, both answering a single question: *where is
the object located relative to a reference camera?* The difference is
**which view the object is visible in**:

1. ``camera_object_cross_mcq`` — cross-view inference
     v_a and v_b are two diverse views of the same scene, the target
     object is visible in v_b but **not** in v_a, and the question asks
     for the object's direction relative to v_a's camera. The model
     therefore has to infer the object's 3D position from v_b and then
     mentally project it into v_a's frame. Answer space covers the full
     8-octant circle (the object can be anywhere around v_a, including
     behind it).

2. ``camera_object_self_mcq`` — single-view grounding
     The target object is visible **in v_a** itself (v_b is kept as
     secondary context so the task signature matches the cross variant,
     but the answer depends only on v_a). The question asks for the
     object's direction relative to the same v_a. Because the object has
     to project into v_a's frustum, the answer is biased toward the
     front hemisphere (Front / Front-Right / Front-Left / Right / Left
     at the edges). To avoid a degenerate "always Front" answer
     distribution, we require the object's bearing |θ| to be at least
     ``self_min_bearing_deg`` (default 30°), so the target sits visibly
     off the optical axis — closer to a 45°/90° layout.

Direction computation (horizontal-plane projection, identical to camera-camera)
-------------------------------------------------------------------------------
All data in this repo is z-up world (ARKitScenes / embodiedscan). We:
  1. Take the world-xy offset ``Δ_xy = (obj - cam).xy``;
  2. Project the reference camera's forward/right basis vectors onto
     world-xy and renormalize to obtain ``fwd_hp`` / ``right_hp``;
  3. Decompose ``Δ_xy = dx · right_hp + dz · fwd_hp``.
This kills pitch/roll leakage and matches the BEV renderer exactly.

Answer space: 8 options — the octant set
{Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left}
permuted into A/B/C/D/E/F/G/H slots. Bearing is θ = atan2(dx, dz).

Filters (configurable via YAML)
-------------------------------
Shared between both sub-tasks:
  * ``min_rot_angle`` / ``min_translation`` — generic 3D pose-diversity
    threshold on (v_a, v_b) (OR-combined);
  * ``min_horizontal_translation`` — hard floor on |Δxy| between the two
    cameras;
  * ``max_vertical_translation`` — hard ceiling on |Δz| between the two
    cameras; same "co-planar cameras" rationale as camera-camera;
  * ``boundary_margin_deg`` — reject bearings within this many degrees
    of a 45°-octant boundary (22.5° / 67.5° / 112.5° / 157.5° …);
  * ``min_object_horizontal_distance`` — reject targets whose |Δxy|
    from the reference camera is below this (unstable bearings).

Sub-task specific:
  * ``self_min_bearing_deg`` — lower bound on |θ| for the self variant,
    default 30°. Forbids near-axis "Front" answers and concentrates
    mass near 45°/90° layouts.
  * ``cross_min_bearing_deg`` — lower bound on |θ| for the cross
    variant, default 0° (full-circle answers OK).

Diagnostic output is prefixed ``[mmsi_cam_obj]``.
"""

import atexit
import math
import random
from collections import Counter

import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes

_TAG = "[mmsi_cam_obj]"

_DIRECTIONS_8 = (
    "Front",        # bearing ≈   0°
    "Front-Right",  # bearing ≈  45°
    "Right",        # bearing ≈  90°
    "Back-Right",   # bearing ≈ 135°
    "Back",         # bearing ≈ 180°
    "Back-Left",    # bearing ≈-135° / 225°
    "Left",         # bearing ≈ -90° / 270°
    "Front-Left",   # bearing ≈ -45° / 315°
)
_LETTERS = "ABCDEFGH"

# ─── Default thresholds ─────────────────────────────────────────────────

_DEFAULT_BOUNDARY_MARGIN_DEG = 5.0
_DEFAULT_MIN_HORIZONTAL_TRANSLATION = 0.3
_DEFAULT_MAX_VERTICAL_TRANSLATION = 0.5
_DEFAULT_MIN_OBJECT_HORIZONTAL_DISTANCE = 0.3
_DEFAULT_MIN_BEARING_DEG_SELF = 30.0
_DEFAULT_MIN_BEARING_DEG_CROSS = 0.0
_DEFAULT_MAX_RETRIES = 60

# Node tag blacklist: large planar structure classes never make sensible
# "target objects" for a direction question. Intentionally a hard-coded
# small set (identical to the original camera-object handler) rather than
# the full YAML ``filter_tags`` list — the latter may include the generic
# "object" tag and would wipe out most candidates.
_STRUCT_TAGS = ("floor", "ceiling", "wall")


def _p(msg):
    """Single print helper — keeps the prefix consistent and flushes."""
    print(f"{_TAG} {msg}", flush=True)


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Camera-Object"

    # Two independent sub-tasks, each with its own quota in the YAML.
    SUB_TASKS = {
        "camera_object_cross_mcq": {
            "default": 1,
            "handler": "_generate_camera_object_cross_mcq",
        },
        "camera_object_self_mcq": {
            "default": 1,
            "handler": "_generate_camera_object_self_mcq",
        },
    }

    # Process-wide counters aggregated across instances / threads.
    _SKIP_COUNTER = Counter()
    _SCENE_COUNTER = Counter()
    _ATEXIT_REGISTERED = False

    def __init__(self, args):
        super().__init__(args)

        # Boundary margin (degrees) — same semantics as in mmsi_camera_camera.
        self._boundary_margin_deg = self._read_float(
            args, "boundary_margin_deg", _DEFAULT_BOUNDARY_MARGIN_DEG,
            lo=0.0, hi=22.4,
        )
        # Horizontal-plane camera translation floor (on v_a / v_b pair).
        self._min_horizontal_translation = self._read_float(
            args, "min_horizontal_translation",
            _DEFAULT_MIN_HORIZONTAL_TRANSLATION, lo=0.0,
        )
        # Vertical camera translation ceiling (on v_a / v_b pair).
        self._max_vertical_translation = self._read_float(
            args, "max_vertical_translation",
            _DEFAULT_MAX_VERTICAL_TRANSLATION, lo=0.0,
        )
        # Minimum horizontal distance from reference camera to the target.
        self._min_object_horizontal_distance = self._read_float(
            args, "min_object_horizontal_distance",
            _DEFAULT_MIN_OBJECT_HORIZONTAL_DISTANCE, lo=0.0,
        )
        # Per-variant bearing lower bound.
        self._self_min_bearing_deg = self._read_float(
            args, "self_min_bearing_deg",
            _DEFAULT_MIN_BEARING_DEG_SELF, lo=0.0, hi=89.0,
        )
        self._cross_min_bearing_deg = self._read_float(
            args, "cross_min_bearing_deg",
            _DEFAULT_MIN_BEARING_DEG_CROSS, lo=0.0, hi=89.0,
        )
        # Retry budget per sub-task call.
        raw_retries = self._read_arg(args, "max_retries", _DEFAULT_MAX_RETRIES)
        try:
            self._max_retries = max(1, int(raw_retries))
        except (TypeError, ValueError):
            self._max_retries = _DEFAULT_MAX_RETRIES

        _p(
            f"init: min_rot={self.min_rot_angle:.1f}°  "
            f"min_trans={self.min_translation:.2f}m  "
            f"min_hp_trans={self._min_horizontal_translation:.2f}m  "
            f"max_vert_trans={self._max_vertical_translation:.2f}m  "
            f"min_obj_hdist={self._min_object_horizontal_distance:.2f}m  "
            f"boundary_margin={self._boundary_margin_deg:.1f}°  "
            f"self_min_bearing={self._self_min_bearing_deg:.1f}°  "
            f"cross_min_bearing={self._cross_min_bearing_deg:.1f}°  "
            f"max_num_views={self.max_num_views}  "
            f"max_retries={self._max_retries}  "
            f"sub_tasks={list(self.SUB_TASKS.keys())}  answer_space=8way"
        )

        if not AnnotationGenerator._ATEXIT_REGISTERED:
            atexit.register(AnnotationGenerator._dump_summary)
            AnnotationGenerator._ATEXIT_REGISTERED = True

    # ─── Small arg-parsing helpers ──────────────────────────────────────

    @staticmethod
    def _read_arg(args, key, default):
        """Pull ``key`` out of ``args`` whether it's a dict or an object."""
        if hasattr(args, "get"):
            return args.get(key, default)
        return getattr(args, key, default)

    @classmethod
    def _read_float(cls, args, key, default, lo=None, hi=None):
        raw = cls._read_arg(args, key, default)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = float(default)
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    # ─── Diagnostic helpers ──────────────────────────────────────────────

    @classmethod
    def _dump_summary(cls):
        total = cls._SCENE_COUNTER.get("total", 0)
        if total == 0:
            return
        ok_cross = cls._SCENE_COUNTER.get("ok_cross", 0)
        ok_self = cls._SCENE_COUNTER.get("ok_self", 0)
        skip = cls._SCENE_COUNTER.get("skip", 0)
        _p(
            f"summary: scene_invocations total={total} "
            f"ok_cross={ok_cross} ok_self={ok_self} skip={skip}"
        )
        if cls._SKIP_COUNTER:
            pairs = ", ".join(f"{k}={v}"
                              for k, v in cls._SKIP_COUNTER.most_common())
            _p(f"summary: skip_reasons -> {pairs}")

    # ─── Direction math (reference-camera horizontal-plane local frame) ─

    @staticmethod
    def _decompose_on_hp(pose_ref, point_world):
        """Return (dx, dz) of ``point_world`` in ``pose_ref``'s horizontal-
        plane local frame. dx = right, dz = forward. Uses world-xy
        projection of pose_ref's forward/right basis so the result is
        immune to pose_ref's pitch/roll — identical convention to the BEV
        renderer and to mmsi_camera_camera._classify_direction.

        Falls back to the raw camera-frame formula when pose_ref aims
        almost straight up/down (degenerate horizontal basis).
        """
        pose_ref = np.asarray(pose_ref, dtype=float)
        point_world = np.asarray(point_world, dtype=float).reshape(-1)
        pw = point_world[:3]

        dxy = pw[:2] - pose_ref[:2, 3]
        dxw, dyw = float(dxy[0]), float(dxy[1])

        R = pose_ref[:3, :3]
        fwd_w = R @ np.array([0.0, 0.0, 1.0])
        right_w = R @ np.array([1.0, 0.0, 0.0])
        fwd_xy = np.array([float(fwd_w[0]), float(fwd_w[1])])
        right_xy = np.array([float(right_w[0]), float(right_w[1])])

        fwd_n = float(np.linalg.norm(fwd_xy))
        right_n = float(np.linalg.norm(right_xy))

        if fwd_n < 1e-6 or right_n < 1e-6:
            # Camera nearly vertical — use full 3D camera-frame fallback.
            pt_h = np.array([pw[0], pw[1], pw[2], 1.0])
            local = np.linalg.inv(pose_ref) @ pt_h
            return float(local[0]), float(local[2])

        fwd_xy /= fwd_n
        right_xy /= right_n
        dx = dxw * float(right_xy[0]) + dyw * float(right_xy[1])
        dz = dxw * float(fwd_xy[0]) + dyw * float(fwd_xy[1])
        return dx, dz

    @staticmethod
    def _classify_octant(dx, dz):
        """Return (label, bearing_deg). Caller is responsible for the
        degeneracy / boundary / min-bearing filters."""
        bearing_deg = math.degrees(math.atan2(dx, dz))  # 0 = Front, +right
        idx = int(((bearing_deg + 22.5) % 360.0) // 45.0)
        return _DIRECTIONS_8[idx], bearing_deg

    @staticmethod
    def _is_octant_unambiguous(dx, dz, margin_deg):
        if margin_deg <= 0.0:
            return True
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            return True
        bearing_deg = math.degrees(math.atan2(dx, dz))
        offset = ((bearing_deg + 22.5) % 45.0) - 22.5
        return abs(offset) <= (22.5 - margin_deg)

    # ─── Shared pair / object filters ───────────────────────────────────

    def _pair_passes_pose_filters(self, pose_a, pose_b):
        """Camera-pair level filters (rot/trans diversity + horizontal
        floor + vertical ceiling). Returns (ok, reason)."""
        if not self._check_pose_diversity(pose_b, [pose_a],
                                          self.min_rot_angle,
                                          self.min_translation):
            return False, "pose_diversity"

        dxy = pose_b[:3, 3] - pose_a[:3, 3]
        hdist = float(math.hypot(dxy[0], dxy[1]))
        if hdist < self._min_horizontal_translation:
            return False, "pair_horizontal_too_close"

        vdist = abs(float(dxy[2]))
        if vdist > self._max_vertical_translation:
            return False, "pair_vertical_too_far"

        return True, None

    def _object_passes_direction_filters(self, pose_ref, obj_world_xyz,
                                         min_bearing_deg):
        """Check object-vs-camera filters: min horizontal distance,
        non-degenerate offset, boundary margin, and min-bearing.

        Returns ``(ok, reason, dx, dz)`` — (dx, dz) are returned even on
        failure so callers can still log them.
        """
        dx, dz = self._decompose_on_hp(pose_ref, obj_world_xyz)

        if math.hypot(dx, dz) < self._min_object_horizontal_distance:
            return False, "object_too_close", dx, dz

        if abs(dx) < 0.05 and abs(dz) < 0.05:
            return False, "degenerate_offset", dx, dz

        if not self._is_octant_unambiguous(dx, dz, self._boundary_margin_deg):
            return False, "ambiguous_octant_boundary", dx, dz

        if min_bearing_deg > 0.0:
            bearing = math.degrees(math.atan2(dx, dz))
            wrapped = (bearing + 180.0) % 360.0 - 180.0   # (-180, 180]
            if abs(wrapped) < min_bearing_deg:
                # Target sits too close to A's optical axis (Front). Skip
                # to avoid concentrating self-variant answers on "Front".
                return False, "bearing_below_min", dx, dz

        return True, None, dx, dz

    # ─── Random sampler shared by both variants ─────────────────────────

    def _sample_pair_with_target(self, graph, target_in_a, min_bearing_deg,
                                 reject):
        """Random (v_a, v_b, node) sampler.

        Args:
            target_in_a (bool): if True, sample a node that IS visible in
                v_a (self variant); if False, sample a node visible in
                v_b but NOT in v_a (cross variant).
            min_bearing_deg (float): minimum |bearing| in the reference
                camera's horizontal-plane frame.
            reject (Counter): updated with the reason each drawn triple
                failed, for diagnostics.

        Returns the first triple that passes every filter, or ``None``.
        """
        view_ids = [vi for vi in graph.views
                    if graph.views[vi].pose is not None]
        if len(view_ids) < 2:
            reject["insufficient_views_with_pose"] += 1
            return None

        nodes_all = [
            n for n in graph.nodes.values()
            if n.box_3d_world is not None and n.tag not in _STRUCT_TAGS
        ]
        if not nodes_all:
            reject["no_valid_nodes"] += 1
            return None

        for _ in range(self._max_retries):
            v_a = random.choice(view_ids)
            pose_a = graph.views[v_a].pose
            nodes = list(nodes_all)
            random.shuffle(nodes)
            for node in nodes:
                app = set(getattr(node, "view_appearances", []) or [])
                if target_in_a:
                    if v_a not in app:
                        continue
                    cand_v_b = [v for v in view_ids if v != v_a]
                else:  # cross: node visible in some v_b but not in v_a
                    if v_a in app:
                        continue
                    cand_v_b = [v for v in app if v != v_a
                                and graph.views[v].pose is not None]
                if not cand_v_b:
                    continue
                v_b = random.choice(cand_v_b)
                pose_b = graph.views[v_b].pose

                ok_pair, why = self._pair_passes_pose_filters(pose_a, pose_b)
                if not ok_pair:
                    reject[why] += 1
                    continue

                ok_obj, why_obj, dx, dz = self._object_passes_direction_filters(
                    pose_a, node.box_3d_world[:3], min_bearing_deg
                )
                if not ok_obj:
                    reject[why_obj] += 1
                    continue

                return {
                    "v_a": v_a, "v_b": v_b, "node": node,
                    "pose_a": np.asarray(pose_a, dtype=float),
                    "pose_b": np.asarray(pose_b, dtype=float),
                    "dx": dx, "dz": dz,
                }

        reject["retries_exhausted"] += 1
        return None

    # ─── QA builder (shared) ─────────────────────────────────────────────

    def _build_qa(self, graph, cand, variant):
        """Assemble the 4-tuple returned to the pipeline. ``variant`` is
        either 'cross' or 'self' and only affects the question wording."""
        v_a = cand["v_a"]
        v_b = cand["v_b"]
        node = cand["node"]
        dx, dz = cand["dx"], cand["dz"]

        answer_direction, _bearing = self._classify_octant(dx, dz)

        options = list(_DIRECTIONS_8)
        random.shuffle(options)
        answer_letter = _LETTERS[options.index(answer_direction)]
        options_str = "Options: " + " ".join(
            [f"{_LETTERS[i]}. {options[i]}" for i in range(len(options))]
        )

        if variant == "cross":
            # Cross-view: object visible only in image 2, asked w.r.t. v_a.
            question = (
                f"In image 2, where is the {node.tag} located relative to "
                "the camera that took image 1? " + options_str
            )
        else:
            # Self: object visible in image 1, asked w.r.t. v_a itself.
            # Image 2 is shown for scene context only.
            question = (
                f"In image 1, where is the {node.tag} located relative to "
                "the camera that took image 1? " + options_str
            )
        prompt = question + " Answer: " + answer_letter

        processed_images = [
            {"bytes": convert_pil_to_bytes(graph.views[v_a].image)},
            {"bytes": convert_pil_to_bytes(graph.views[v_b].image)},
        ]
        cog_ctx = self._make_cog_context(
            view_indices=[v_a, v_b],
            node_ids=[node.node_id],
            anchor_node_id=node.node_id,
        )
        return prompt, processed_images, QuestionType.MCQ, cog_ctx

    # ─── Sub-task handlers ───────────────────────────────────────────────

    def _generate_camera_object_cross_mcq(self, graph):
        """Cross-view inference: object is in v_b only."""
        AnnotationGenerator._SCENE_COUNTER["total"] += 1
        reject = Counter()
        cand = self._sample_pair_with_target(
            graph,
            target_in_a=False,
            min_bearing_deg=self._cross_min_bearing_deg,
            reject=reject,
        )
        if cand is None:
            for k, v in reject.items():
                AnnotationGenerator._SKIP_COUNTER[f"cross/{k}"] += v
            AnnotationGenerator._SCENE_COUNTER["skip"] += 1
            return None
        AnnotationGenerator._SCENE_COUNTER["ok_cross"] += 1
        return self._build_qa(graph, cand, variant="cross")

    def _generate_camera_object_self_mcq(self, graph):
        """Single-view grounding: object is visible in v_a."""
        AnnotationGenerator._SCENE_COUNTER["total"] += 1
        reject = Counter()
        cand = self._sample_pair_with_target(
            graph,
            target_in_a=True,
            min_bearing_deg=self._self_min_bearing_deg,
            reject=reject,
        )
        if cand is None:
            for k, v in reject.items():
                AnnotationGenerator._SKIP_COUNTER[f"self/{k}"] += v
            AnnotationGenerator._SCENE_COUNTER["skip"] += 1
            return None
        AnnotationGenerator._SCENE_COUNTER["ok_self"] += 1
        return self._build_qa(graph, cand, variant="self")
