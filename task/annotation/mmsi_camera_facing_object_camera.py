"""
MMSI-Bench: Camera–Facing-Object–Camera direction task.

Question template (8-way MCQ):
    "Standing at the camera that took image 1 and facing the {object},
     in which direction is the camera that took image 2?"

The task is a direction-in-octants variant of the Camera-Camera MMSI task,
but with a twist: the reference "forward" direction is NOT image 1's
camera optical axis — instead it is the 2D horizontal vector pointing
from Camera A toward the target object. This probes whether the model
can (a) localize an object visible in image 1, (b) reorient its mental
body frame to face that object, and (c) answer a subsequent direction
question about a second camera in that reoriented frame.

Geometry (horizontal-plane, z-up world)
---------------------------------------
All data in this repo is z-up world (ARKitScenes / embodiedscan / …), so
the horizontal plane is world-xy. We construct a fresh 2D local frame at
Camera A's location with "forward" aiming at the object:

    fwd_hp   = normalize( (obj.xy - cam_a.xy) )
    right_hp = (fwd_hp.y, -fwd_hp.x)          # fwd rotated -90° (clockwise)

Then decompose Camera B's horizontal offset in that frame:

    Δ_xy = (cam_b - cam_a).xy
    dz   = Δ_xy · fwd_hp        (+ = "in front of" the virtual heading)
    dx   = Δ_xy · right_hp      (+ = "to the right" of the virtual heading)

Bearing θ = atan2(dx, dz) is binned into eight 45°-wide octants centered
on 0°, ±45°, ±90°, ±135°, 180°. Pairs whose bearing lies within
``boundary_margin_deg`` of an octant boundary (22.5°, 67.5°, 112.5°,
157.5°, …) are rejected as ambiguous.

Filters
-------
* Camera-pair level (same as mmsi_camera_camera):
  - generic 3D pose diversity (min_rot_angle OR min_translation);
  - horizontal-plane translation floor (min_horizontal_translation);
  - vertical translation ceiling (max_vertical_translation).
* Object level:
  - must NOT be floor / ceiling / wall (structural planes — not sensible
    targets to "face");
  - must be visible in image 1 (default variant; so the "Standing at A
    and facing X" instruction is visually verifiable from image 1);
  - horizontal distance from Camera A to the object must be at least
    ``min_object_horizontal_distance`` (avoids unstable virtual headings
    when the target is almost on top of the camera);
  - the virtual heading (A→object) must differ from A's real camera
    forward by at least ``min_facing_deviation_deg`` (avoids degenerate
    cases where the task reduces to plain camera-camera; set to 0 to
    disable this filter).
* Answer-level:
  - octant-boundary margin (boundary_margin_deg), as above.

Frame-selection policy
----------------------
Same B1 enumerate-then-sample pattern as mmsi_camera_camera: we
enumerate every legal (object_node, v_a, v_b) triple, then diversely
sample up to ``sub_tasks.camera_facing_object_camera_mcq`` items per
scene with round-robin over distinct object anchors and distinct
view-pairs.

Diagnostic output is prefixed ``[mmsi_cam_face_obj_cam]``.
"""

import atexit
import math
import random
from collections import Counter, defaultdict

import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes

_TAG = "[mmsi_cam_face_obj_cam]"

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
_DEFAULT_MIN_FACING_DEVIATION_DEG = 20.0

# Structural tags that never make sensible "facing" targets. Kept as a
# small hard-coded set even when the YAML filter_tags is looser — you
# cannot meaningfully "face the floor" in the task semantics.
_STRUCT_TAGS = ("floor", "ceiling", "wall")


def _p(msg):
    """Single print helper — keeps the prefix consistent and flushes."""
    print(f"{_TAG} {msg}", flush=True)


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Camera-FacingObject-Camera"
    SUB_TASKS = {
        # ``default=1`` means: call handler once per scene. The handler
        # itself returns a list of up to ``count`` QA tuples, so setting
        # ``sub_tasks.camera_facing_object_camera_mcq: N`` yields up to
        # N QAs per scene.
        "camera_facing_object_camera_mcq": {
            "default": 1,
            "handler": "_generate_camera_facing_object_camera_mcq",
        },
    }

    # Process-wide counters aggregated across instances / threads.
    _SKIP_COUNTER = Counter()
    _SCENE_COUNTER = Counter()
    _ATEXIT_REGISTERED = False

    def __init__(self, args):
        super().__init__(args)

        self._qa_quota = self.get_sub_task_count(
            "camera_facing_object_camera_mcq", default=1
        )

        self._boundary_margin_deg = self._read_float(
            args, "boundary_margin_deg", _DEFAULT_BOUNDARY_MARGIN_DEG,
            lo=0.0, hi=22.4,
        )
        self._min_horizontal_translation = self._read_float(
            args, "min_horizontal_translation",
            _DEFAULT_MIN_HORIZONTAL_TRANSLATION, lo=0.0,
        )
        self._max_vertical_translation = self._read_float(
            args, "max_vertical_translation",
            _DEFAULT_MAX_VERTICAL_TRANSLATION, lo=0.0,
        )
        self._min_object_horizontal_distance = self._read_float(
            args, "min_object_horizontal_distance",
            _DEFAULT_MIN_OBJECT_HORIZONTAL_DISTANCE, lo=0.0,
        )
        # If the virtual heading (A→object) is too close to A's real
        # camera forward, the task collapses to plain camera-camera and
        # loses its "reorient your body to face X" value. Reject such
        # triples. 0 disables the filter.
        self._min_facing_deviation_deg = self._read_float(
            args, "min_facing_deviation_deg",
            _DEFAULT_MIN_FACING_DEVIATION_DEG, lo=0.0, hi=179.0,
        )

        _p(
            f"init: min_rot={self.min_rot_angle:.1f}°  "
            f"min_trans={self.min_translation:.2f}m  "
            f"min_hp_trans={self._min_horizontal_translation:.2f}m  "
            f"max_vert_trans={self._max_vertical_translation:.2f}m  "
            f"min_obj_hdist={self._min_object_horizontal_distance:.2f}m  "
            f"boundary_margin={self._boundary_margin_deg:.1f}°  "
            f"min_facing_dev={self._min_facing_deviation_deg:.1f}°  "
            f"max_num_views={self.max_num_views}  "
            f"quota_per_scene={self._qa_quota}  "
            f"policy=enumerate_then_sample  answer_space=8way  "
            f"object_visibility=image1_only"
        )

        if not AnnotationGenerator._ATEXIT_REGISTERED:
            atexit.register(AnnotationGenerator._dump_summary)
            AnnotationGenerator._ATEXIT_REGISTERED = True

    # ─── Small arg-parsing helpers ──────────────────────────────────────

    @staticmethod
    def _read_arg(args, key, default):
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
    def _record_skip(cls, reason):
        cls._SKIP_COUNTER[reason] += 1

    @classmethod
    def _dump_summary(cls):
        total = cls._SCENE_COUNTER.get("total", 0)
        if total == 0:
            return
        ok = cls._SCENE_COUNTER.get("ok", 0)
        skip = cls._SCENE_COUNTER.get("skip", 0)
        qa_ok = cls._SCENE_COUNTER.get("qa_ok", 0)
        _p(
            f"summary: scenes total={total} ok={ok} skip={skip}  "
            f"qa_ok={qa_ok}  "
            f"(skip_ratio={100.0 * skip / max(total, 1):.1f}%)"
        )
        if cls._SKIP_COUNTER:
            pairs = ", ".join(f"{k}={v}"
                              for k, v in cls._SKIP_COUNTER.most_common())
            _p(f"summary: skip_reasons -> {pairs}")

    @staticmethod
    def _scene_id(graph):
        raw = getattr(graph, "raw_example", {}) or {}
        for k in ("scene_id", "scene", "scan_id", "sample_id"):
            v = raw.get(k)
            if v is not None:
                return str(v)
        return "?"

    @staticmethod
    def _preview(text, n=120):
        if not isinstance(text, str):
            return str(text)
        text = text.replace("\n", " ").strip()
        return text if len(text) <= n else text[:n - 1] + "…"

    # ─── Geometry helpers ───────────────────────────────────────────────

    @staticmethod
    def _camera_forward_xy(pose):
        """Project the OpenCV-convention camera +Z axis onto world-xy.

        Returns the unit xy vector, or None if the camera is nearly
        vertical (no meaningful horizontal forward).
        """
        R = np.asarray(pose, dtype=float)[:3, :3]
        fwd_world = R @ np.array([0.0, 0.0, 1.0])
        vx, vy = float(fwd_world[0]), float(fwd_world[1])
        n = math.hypot(vx, vy)
        if n < 1e-6:
            return None
        return np.array([vx / n, vy / n])

    @staticmethod
    def _decompose_in_facing_frame(pose_a, pose_b, obj_xyz):
        """Decompose B's horizontal offset in A's "face-the-object" frame.

        Constructs a 2D local frame at Camera A with:
            fwd_hp   = normalize( (obj.xy - cam_a.xy) )
            right_hp = (fwd_hp.y, -fwd_hp.x)      # fwd rotated -90° (CW)

        and returns (dx, dz, fwd_hp, right_hp, facing_ok). If the object
        is horizontally on top of Camera A (|Δxy| < 1e-6), ``facing_ok``
        is False and (dx, dz) is (nan, nan).
        """
        pose_a = np.asarray(pose_a, dtype=float)
        pose_b = np.asarray(pose_b, dtype=float)
        obj = np.asarray(obj_xyz, dtype=float).reshape(-1)[:3]

        cam_a_xy = pose_a[:2, 3]
        cam_b_xy = pose_b[:2, 3]

        fwd_vec = obj[:2] - cam_a_xy
        fwd_n = float(np.linalg.norm(fwd_vec))
        if fwd_n < 1e-6:
            return float("nan"), float("nan"), None, None, False
        fwd_hp = fwd_vec / fwd_n
        # Rotate forward by -90° (clockwise) to get the right vector,
        # so that (right_hp, fwd_hp) forms a right-handed 2D basis
        # consistent with the "+x right, +z forward" convention used
        # by the octant labels and the BEV renderer.
        right_hp = np.array([float(fwd_hp[1]), -float(fwd_hp[0])])

        dxy = cam_b_xy - cam_a_xy
        dz = float(dxy[0] * fwd_hp[0] + dxy[1] * fwd_hp[1])
        dx = float(dxy[0] * right_hp[0] + dxy[1] * right_hp[1])
        return dx, dz, fwd_hp, right_hp, True

    @staticmethod
    def _classify_octant(dx, dz):
        """Return (label_or_None, bearing_deg).

        Degenerate offsets (|dx| < 5cm and |dz| < 5cm) return label=None.
        """
        if abs(dx) < 0.05 and abs(dz) < 0.05:
            return None, math.degrees(math.atan2(dx, dz))
        bearing_deg = math.degrees(math.atan2(dx, dz))
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

    # ─── Candidate enumeration & diverse sampling ────────────────────────

    def _enumerate_candidates(self, graph):
        """Enumerate every legal (object_node, v_a, v_b) triple.

        Returns (candidates, reject_counter).
        """
        candidates = []
        reject = Counter()

        # All posed views, indexed once.
        posed_views = [vi for vi in graph.views
                       if graph.views[vi].pose is not None]
        if len(posed_views) < 2:
            reject["insufficient_posed_views"] += 1
            return candidates, reject

        # Candidate target objects: visible in at least one view, not
        # structural, has a 3D world box.
        for nid, node in graph.nodes.items():
            if node.box_3d_world is None:
                continue
            if node.tag in _STRUCT_TAGS:
                continue
            obj_xyz = np.asarray(node.box_3d_world, dtype=float).reshape(-1)[:3]

            # Views in which the object is visible (candidates for v_a).
            app = [vi for vi in (node.view_appearances or {})
                   if graph.views.get(vi) is not None
                   and graph.views[vi].pose is not None]
            if len(app) < 1:
                continue

            # All posed views are candidates for v_b (object visibility
            # in v_b is NOT required by the self variant; the question
            # only needs image 1 to visually anchor "the object").
            for v_a in app:
                pose_a = np.asarray(graph.views[v_a].pose, dtype=float)

                # Horizontal distance from A to object: reject near-zero.
                if self._min_object_horizontal_distance > 0.0:
                    hdist_ao = float(math.hypot(
                        float(obj_xyz[0] - pose_a[0, 3]),
                        float(obj_xyz[1] - pose_a[1, 3]),
                    ))
                    if hdist_ao < self._min_object_horizontal_distance:
                        reject["object_too_close_to_cam_a"] += 1
                        continue

                # Virtual forward direction (A → object, projected on xy).
                fwd_vec = obj_xyz[:2] - pose_a[:2, 3]
                fwd_n = float(np.linalg.norm(fwd_vec))
                if fwd_n < 1e-6:
                    reject["degenerate_facing_direction"] += 1
                    continue
                fwd_hp = fwd_vec / fwd_n

                # Reject if virtual heading is too close to A's real
                # camera forward (task collapses to cam-cam otherwise).
                if self._min_facing_deviation_deg > 0.0:
                    cam_fwd_xy = self._camera_forward_xy(pose_a)
                    if cam_fwd_xy is not None:
                        cos_ang = float(np.clip(
                            fwd_hp[0] * cam_fwd_xy[0]
                            + fwd_hp[1] * cam_fwd_xy[1],
                            -1.0, 1.0,
                        ))
                        deviation_deg = math.degrees(math.acos(cos_ang))
                        if deviation_deg < self._min_facing_deviation_deg:
                            reject["facing_too_close_to_cam_fwd"] += 1
                            continue

                for v_b in posed_views:
                    if v_b == v_a:
                        continue
                    pose_b = np.asarray(graph.views[v_b].pose, dtype=float)

                    # Camera-pair pose diversity (OR, 3D).
                    if not self._check_pose_diversity(
                        pose_b, [pose_a],
                        self.min_rot_angle, self.min_translation,
                    ):
                        reject["pose_not_diverse"] += 1
                        continue

                    # Horizontal-plane floor (AND, 2D).
                    dxy_pair = pose_b[:3, 3] - pose_a[:3, 3]
                    if self._min_horizontal_translation > 0.0:
                        hp_dist = float(math.hypot(
                            float(dxy_pair[0]), float(dxy_pair[1])
                        ))
                        if hp_dist < self._min_horizontal_translation:
                            reject["horizontal_too_close"] += 1
                            continue

                    # Vertical ceiling (AND, 1D).
                    if self._max_vertical_translation > 0.0:
                        if abs(float(dxy_pair[2])) > self._max_vertical_translation:
                            reject["vertical_too_far"] += 1
                            continue

                    dx, dz, _, _, ok_geom = \
                        self._decompose_in_facing_frame(pose_a, pose_b, obj_xyz)
                    if not ok_geom:
                        reject["degenerate_facing_direction"] += 1
                        continue

                    label, _bearing = self._classify_octant(dx, dz)
                    if label is None:
                        reject["degenerate_direction"] += 1
                        continue

                    if not self._is_octant_unambiguous(
                        dx, dz, self._boundary_margin_deg
                    ):
                        reject["ambiguous_octant_boundary"] += 1
                        continue

                    candidates.append({
                        "node": node,
                        "v_a": v_a,
                        "v_b": v_b,
                        "pose_a": pose_a,
                        "pose_b": pose_b,
                        "obj_xyz": obj_xyz,
                        "answer": label,
                        "dx": dx,
                        "dz": dz,
                    })
        return candidates, reject

    @staticmethod
    def _diverse_sample(candidates, n):
        """Round-robin diverse sample over distinct (node, view-pair) keys.

        Same spirit as mmsi_camera_camera._diverse_sample: spread picks
        across anchor objects first, then across distinct view pairs.
        """
        if n <= 0 or not candidates:
            return []
        if n >= len(candidates):
            out = list(candidates)
            random.shuffle(out)
            return out

        by_anchor = defaultdict(lambda: defaultdict(list))
        for c in candidates:
            anchor_id = c["node"].node_id
            pair_key = (min(c["v_a"], c["v_b"]), max(c["v_a"], c["v_b"]))
            by_anchor[anchor_id][pair_key].append(c)

        anchor_ids = list(by_anchor.keys())
        random.shuffle(anchor_ids)
        for aid in anchor_ids:
            pair_keys = list(by_anchor[aid].keys())
            random.shuffle(pair_keys)
            by_anchor[aid] = {pk: by_anchor[aid][pk] for pk in pair_keys}
            for pk in by_anchor[aid]:
                random.shuffle(by_anchor[aid][pk])

        picked = []
        while len(picked) < n:
            progressed = False
            for aid in list(anchor_ids):
                buckets = by_anchor[aid]
                if not buckets:
                    anchor_ids.remove(aid)
                    continue
                chosen_pk = None
                for pk in list(buckets.keys()):
                    if buckets[pk]:
                        chosen_pk = pk
                        break
                    else:
                        del buckets[pk]
                if chosen_pk is None:
                    anchor_ids.remove(aid)
                    continue
                picked.append(buckets[chosen_pk].pop())
                progressed = True
                if len(picked) == n:
                    break
            if not progressed:
                break
        return picked

    # ─── QA builder (single candidate) ───────────────────────────────────

    def _build_one_qa(self, graph, cand):
        node = cand["node"]
        v_a, v_b = cand["v_a"], cand["v_b"]
        answer_direction = cand["answer"]

        options = list(_DIRECTIONS_8)
        random.shuffle(options)
        answer_letter = _LETTERS[options.index(answer_direction)]
        options_str = "Options: " + " ".join(
            [f"{_LETTERS[i]}. {options[i]}" for i in range(len(options))]
        )
        question = (
            f"Standing at the camera that took image 1 and facing the "
            f"{node.tag}, in which direction is the camera that took "
            f"image 2? " + options_str
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
        # Reasoning overlay — consumed by CognitiveMapRenderer to draw
        # the "facing the object" 8-sector wedge around Camera A,
        # together with the A→object virtual-heading line and a star
        # at the target object. We pass the *virtual* forward/right
        # (A→object on world-xy, and forward rotated clockwise 90°)
        # explicitly so the renderer does NOT fall back to A's real
        # optical axis (which would contradict the task semantics).
        if cog_ctx is not None:
            pose_a = cand["pose_a"]
            pose_b = cand["pose_b"]
            obj_xyz = cand["obj_xyz"]

            a_wxy = (float(pose_a[0, 3]), float(pose_a[1, 3]))
            b_wxy = (float(pose_b[0, 3]), float(pose_b[1, 3]))
            obj_wxy = (float(obj_xyz[0]), float(obj_xyz[1]))

            # Virtual heading: A → object on world-xy.
            vf_x = obj_wxy[0] - a_wxy[0]
            vf_y = obj_wxy[1] - a_wxy[1]
            vf_n = math.hypot(vf_x, vf_y) or 1.0
            vf_x /= vf_n
            vf_y /= vf_n
            # Right = forward rotated clockwise 90° on xy — matches the
            # local frame in ``_decompose_in_facing_frame``.
            vr_x, vr_y = vf_y, -vf_x

            virtual_yaw_deg = float(math.degrees(math.atan2(vf_y, vf_x)))

            cog_ctx.extra["reasoning_overlay"] = {
                "kind": "mmsi_cam_face_obj_cam",
                "anchor_view_idx": int(v_a),
                "target_view_idx": int(v_b),
                "dx": float(cand["dx"]),
                "dz": float(cand["dz"]),
                "answer": str(answer_direction),
                # World-frame horizontal-plane (xy) info
                "a_world_xy": [a_wxy[0], a_wxy[1]],
                "b_world_xy": [b_wxy[0], b_wxy[1]],
                "delta_world_xy": [b_wxy[0] - a_wxy[0], b_wxy[1] - a_wxy[1]],
                "a_yaw_world_deg": virtual_yaw_deg,
                # Target-object info (what "facing" refers to).
                "object_world_xy": [obj_wxy[0], obj_wxy[1]],
                "facing_object_tag": str(node.tag),
                # SceneNode.node_id is a *string* (see scene_graph.py):
                #   singleview → f"{obj_idx}"
                #   multiview  → str(box_3d)   ← 9-dim float list repr
                # So we must NOT cast to int here — multiview IDs are
                # never decimal integers and will raise ValueError.
                "facing_object_node_id": str(node.node_id),
                # Virtual basis (unit world-xy vectors) — overrides A's
                # real optical axis inside the overlay drawer.
                "virtual_fwd_xy":   [float(vf_x), float(vf_y)],
                "virtual_right_xy": [float(vr_x), float(vr_y)],
                "virtual_yaw_world_deg": virtual_yaw_deg,
            }
        return (prompt, processed_images, QuestionType.MCQ, cog_ctx,
                options, answer_letter)

    # ─── Handler ─────────────────────────────────────────────────────────

    def _generate_camera_facing_object_camera_mcq(self, graph):
        AnnotationGenerator._SCENE_COUNTER["total"] += 1
        sid = self._scene_id(graph)

        n_views = len(graph.views)
        n_poses = sum(1 for vi in graph.views
                      if graph.views[vi].pose is not None)
        n_nodes = len(graph.nodes)
        _p(
            f"scene={sid} entry: views={n_views} (w/pose={n_poses}) "
            f"nodes={n_nodes}"
        )

        candidates, reject = self._enumerate_candidates(graph)
        _p(
            f"scene={sid} enum: candidates={len(candidates)}  "
            f"rejects={dict(reject) if reject else '{}'}"
        )

        if not candidates:
            reason = "no_candidates"
            AnnotationGenerator._record_skip(reason)
            AnnotationGenerator._SCENE_COUNTER["skip"] += 1
            _p(
                f"scene={sid} views={n_views} poses={n_poses} "
                f"nodes={n_nodes} status=SKIP reason={reason}"
            )
            return None

        quota = max(int(self._qa_quota), 1)
        sampled = self._diverse_sample(candidates, quota)
        distinct_anchors = len({c["node"].node_id for c in sampled})
        distinct_pairs = len({(min(c["v_a"], c["v_b"]),
                                max(c["v_a"], c["v_b"])) for c in sampled})
        _p(
            f"scene={sid} sampled: quota={quota} pool={len(candidates)} "
            f"taken={len(sampled)} distinct_anchors={distinct_anchors} "
            f"distinct_pairs={distinct_pairs}"
        )

        results = []
        for k, cand in enumerate(sampled):
            (prompt, processed_images, qtype, cog_ctx,
             options, answer_letter) = self._build_one_qa(graph, cand)
            v_a, v_b = cand["v_a"], cand["v_b"]
            node = cand["node"]
            _p(
                f"scene={sid} qa[{k}]: anchor={node.tag}({node.node_id}) "
                f"pair=({v_a},{v_b}) dx={cand['dx']:+.3f} "
                f"dz={cand['dz']:+.3f} -> {cand['answer']} "
                f"ans={answer_letter}  options={options}  "
                f"prompt={self._preview(prompt, 140)}"
            )
            results.append((prompt, processed_images, qtype, cog_ctx))

        if not results:
            reason = "sample_empty"
            AnnotationGenerator._record_skip(reason)
            AnnotationGenerator._SCENE_COUNTER["skip"] += 1
            _p(
                f"scene={sid} views={n_views} poses={n_poses} "
                f"nodes={n_nodes} status=SKIP reason={reason}"
            )
            return None

        AnnotationGenerator._SCENE_COUNTER["ok"] += 1
        AnnotationGenerator._SCENE_COUNTER["qa_ok"] += len(results)
        _p(
            f"scene={sid} views={n_views} poses={n_poses} "
            f"nodes={n_nodes} status=OK qa_generated={len(results)} "
            f"pool={len(candidates)}"
        )
        return results
