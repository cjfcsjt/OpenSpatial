"""
MMSI-Bench: Object-Facing-Object-Object direction task (multi-view).

Question template (8-way MCQ):
    "In image 1 you can see the {anchor_object}. In image 2 you can see
     the {orienting_object}. In image 3 you can see the {query_object}.
     If I stand at the {anchor_object} and face the {orienting_object},
     in which direction is the {query_object}?"

Semantics
---------
This is the **multi-view** variant: each of the three objects is only
visible in its own image, so the model must fuse information across
three views to (a) identify each object, (b) mentally erect a local
frame at the anchor with forward aimed at the orienting object, and
(c) locate the query object in that frame.

Geometry is purely world-frame and identical to the single-view
version: we build a 2D local frame at the anchor object's world-xy
location, with "+Z (forward)" pointing toward the orienting object and
"+X (right)" obtained by rotating forward -90° (clockwise). The query
object's xy offset is decomposed in that frame and binned into one of
eight 45°-wide octants centered on 0° (Front), ±45°, ±90°, ±135°, 180°.

    fwd_hp   = normalize( orienting.xy − anchor.xy )
    right_hp = (fwd_hp.y, -fwd_hp.x)           # clockwise 90°

    Δ_xy = query.xy − anchor.xy
    dz   = Δ_xy · fwd_hp        (+ = "in front of" orienting direction)
    dx   = Δ_xy · right_hp      (+ = "to the right" of orienting direction)

Filters
-------
* Visibility (cross-view, exclusive):
  - anchor is visible in view v_a but NOT in v_o and NOT in v_q;
  - orienting is visible in view v_o but NOT in v_a and NOT in v_q;
  - query is visible in view v_q but NOT in v_a and NOT in v_o;
  - v_a, v_o, v_q are three distinct posed views.
* Tag distinctness: anchor / orienting / query must have three
  different ``tag`` values.
* Structural tags (floor / ceiling / wall) are excluded for every role.
* Geometry stability:
  - ``min_anchor_to_orienting_distance``,
    ``min_anchor_to_query_distance``,
    ``min_orienting_to_query_distance``.
* Pose diversity: the three selected views must be pairwise diverse
  (OR-combined rotation / translation), same rule as other multi-view
  MMSI tasks.
* Answer-level: ``boundary_margin_deg`` rejects bearings within ± that
  margin of an octant boundary (22.5° / 67.5° / 112.5° / 157.5°).

Frame-selection policy
----------------------
Enumerate every legal (anchor_node, orienting_node, query_node,
v_a, v_o, v_q) sextuple, then diversely sample up to
``sub_tasks.object_facing_object_object_mcq`` items per scene,
round-robin over distinct anchor objects and distinct (anchor, orienting)
pairs so one object pair does not dominate the quota.

Diagnostic output is prefixed ``[mmsi_obj_face_obj_obj]``.
"""

import atexit
import math
import random
from collections import Counter, defaultdict

import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes

_TAG = "[mmsi_obj_face_obj_obj]"

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

# ─── Default thresholds ────────────────────────────────────────────────
_DEFAULT_BOUNDARY_MARGIN_DEG = 5.0
_DEFAULT_MIN_ANCHOR_TO_ORIENTING_DISTANCE = 0.3
_DEFAULT_MIN_ANCHOR_TO_QUERY_DISTANCE = 0.3
_DEFAULT_MIN_ORIENTING_TO_QUERY_DISTANCE = 0.1

# Structural tags that never make sensible scene-reference objects.
_STRUCT_TAGS = ("floor", "ceiling", "wall")

def _p(msg):
    print(f"{_TAG} {msg}", flush=True)

class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Object-FacingObject-Object"
    SUB_TASKS = {
        # ``default=1`` means: call handler once per scene. The handler
        # returns up to ``count`` QA tuples, so setting
        # ``sub_tasks.object_facing_object_object_mcq: N`` yields up to
        # N QAs per scene.
        "object_facing_object_object_mcq": {
            "default": 1,
            "handler": "_generate_object_facing_object_object_mcq",
        },
    }

    # Process-wide counters aggregated across instances / threads.
    _SKIP_COUNTER = Counter()
    _SCENE_COUNTER = Counter()
    _ATEXIT_REGISTERED = False

    def __init__(self, args):
        super().__init__(args)

        self._qa_quota = self.get_sub_task_count(
            "object_facing_object_object_mcq", default=1
        )

        self._boundary_margin_deg = self._read_float(
            args, "boundary_margin_deg", _DEFAULT_BOUNDARY_MARGIN_DEG,
            lo=0.0, hi=22.4,
        )
        self._min_anchor_to_orienting_distance = self._read_float(
            args, "min_anchor_to_orienting_distance",
            _DEFAULT_MIN_ANCHOR_TO_ORIENTING_DISTANCE, lo=0.0,
        )
        self._min_anchor_to_query_distance = self._read_float(
            args, "min_anchor_to_query_distance",
            _DEFAULT_MIN_ANCHOR_TO_QUERY_DISTANCE, lo=0.0,
        )
        self._min_orienting_to_query_distance = self._read_float(
            args, "min_orienting_to_query_distance",
            _DEFAULT_MIN_ORIENTING_TO_QUERY_DISTANCE, lo=0.0,
        )

        _p(
            f"init: min_anchor_to_orienting={self._min_anchor_to_orienting_distance:.2f}m  "
            f"min_anchor_to_query={self._min_anchor_to_query_distance:.2f}m  "
            f"min_orienting_to_query={self._min_orienting_to_query_distance:.2f}m  "
            f"boundary_margin={self._boundary_margin_deg:.1f}°  "
            f"min_rot={self.min_rot_angle:.1f}°  "
            f"min_trans={self.min_translation:.2f}m  "
            f"max_num_views={self.max_num_views}  "
            f"quota_per_scene={self._qa_quota}  "
            f"policy=enumerate_then_sample  answer_space=8way  "
            f"visibility=three_exclusive_views"
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

    # ─── Diagnostic helpers ─────────────────────────────────────────────

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
    def _decompose_in_object_facing_frame(anchor_xy, orienting_xy, query_xy):
        """Decompose query.xy offset in the (anchor, facing-orienting) frame.

            fwd_hp   = normalize( orienting.xy − anchor.xy )
            right_hp = (fwd_hp.y, -fwd_hp.x)      # fwd rotated -90° (CW)

            dz       = (query − anchor) · fwd_hp
            dx       = (query − anchor) · right_hp

        Returns (dx, dz, fwd_hp, right_hp, ok). ``ok`` is False when the
        facing vector is degenerate (|orienting − anchor| < 1e-6).
        """
        a = np.asarray(anchor_xy, dtype=float).reshape(-1)[:2]
        o = np.asarray(orienting_xy, dtype=float).reshape(-1)[:2]
        q = np.asarray(query_xy, dtype=float).reshape(-1)[:2]

        fwd_vec = o - a
        fwd_n = float(np.linalg.norm(fwd_vec))
        if fwd_n < 1e-6:
            return float("nan"), float("nan"), None, None, False
        fwd_hp = fwd_vec / fwd_n
        # Rotate forward by -90° (clockwise) to get the right vector.
        right_hp = np.array([float(fwd_hp[1]), -float(fwd_hp[0])])

        dxy = q - a
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

    # ─── Candidate enumeration & diverse sampling ───────────────────────

    def _enumerate_candidates(self, graph):
        """Enumerate every legal (anchor, orienting, query, v_a, v_o, v_q).

        Multi-view requirements:
          * anchor/orienting/query are three distinct non-structural
            nodes with a 3D world box, and three distinct tags;
          * each object is visible in its own assigned view (v_a, v_o,
            v_q) and NOT visible in the other two views;
          * v_a, v_o, v_q are three distinct posed views, pairwise
            pose-diverse;
          * horizontal distance floors between the three objects hold;
          * the resulting bearing is not within ``boundary_margin_deg``
            of an octant boundary.
        """
        candidates = []
        reject = Counter()

        posed_views = [vi for vi in graph.views
                       if graph.views[vi].pose is not None]
        if len(posed_views) < 3:
            reject["insufficient_posed_views"] += 1
            return candidates, reject

        # Eligible nodes: not structural, have a 3D world box.
        obj_nodes = [n for n in graph.nodes.values()
                     if n.box_3d_world is not None
                     and n.tag not in _STRUCT_TAGS]
        if len(obj_nodes) < 3:
            reject["fewer_than_three_objects"] += 1
            return candidates, reject

        # Pre-compute visibility sets (as sets of posed-view indices).
        posed_set = set(posed_views)
        vis_sets = {}
        for n in obj_nodes:
            s = set()
            for vi in (n.view_appearances or {}):
                if vi in posed_set:
                    s.add(vi)
            vis_sets[n.node_id] = s

        # Cache poses once — diversity check runs on every sextuple.
        pose_cache = {vi: np.asarray(graph.views[vi].pose, dtype=float)
                      for vi in posed_views}

        # Triple-loop over (anchor, orienting, query) with distinct
        # node ids AND distinct tags.
        for a_node in obj_nodes:
            a_xy = np.asarray(a_node.box_3d_world, dtype=float
                              ).reshape(-1)[:2]
            vis_a = vis_sets[a_node.node_id]
            if not vis_a:
                continue

            for o_node in obj_nodes:
                if o_node.node_id == a_node.node_id:
                    continue
                if o_node.tag == a_node.tag:
                    reject["duplicate_tag_a_o"] += 1
                    continue
                o_xy = np.asarray(o_node.box_3d_world, dtype=float
                                  ).reshape(-1)[:2]

                # anchor→orienting distance floor.
                if self._min_anchor_to_orienting_distance > 0.0:
                    d_ao = float(math.hypot(
                        float(o_xy[0] - a_xy[0]),
                        float(o_xy[1] - a_xy[1]),
                    ))
                    if d_ao < self._min_anchor_to_orienting_distance:
                        reject["anchor_orienting_too_close"] += 1
                        continue

                vis_o = vis_sets[o_node.node_id]
                if not vis_o:
                    continue

                for q_node in obj_nodes:
                    if q_node.node_id in (a_node.node_id, o_node.node_id):
                        continue
                    if q_node.tag in (a_node.tag, o_node.tag):
                        reject["duplicate_tag_with_q"] += 1
                        continue
                    q_xy = np.asarray(q_node.box_3d_world, dtype=float
                                      ).reshape(-1)[:2]

                    # anchor→query distance floor.
                    if self._min_anchor_to_query_distance > 0.0:
                        d_aq = float(math.hypot(
                            float(q_xy[0] - a_xy[0]),
                            float(q_xy[1] - a_xy[1]),
                        ))
                        if d_aq < self._min_anchor_to_query_distance:
                            reject["anchor_query_too_close"] += 1
                            continue

                    # orienting→query distance floor (keeps query
                    # distinct from the facing target).
                    if self._min_orienting_to_query_distance > 0.0:
                        d_oq = float(math.hypot(
                            float(q_xy[0] - o_xy[0]),
                            float(q_xy[1] - o_xy[1]),
                        ))
                        if d_oq < self._min_orienting_to_query_distance:
                            reject["orienting_query_too_close"] += 1
                            continue

                    dx, dz, _, _, ok_geom = \
                        self._decompose_in_object_facing_frame(
                            a_xy, o_xy, q_xy)
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

                    vis_q = vis_sets[q_node.node_id]
                    if not vis_q:
                        continue

                    # Build **exclusive** per-object candidate view sets.
                    #   v_a must show anchor but NEITHER orienting NOR
                    #   query. Mirror for v_o / v_q.
                    only_a = vis_a - vis_o - vis_q
                    only_o = vis_o - vis_a - vis_q
                    only_q = vis_q - vis_a - vis_o
                    if not only_a or not only_o or not only_q:
                        reject["no_exclusive_views"] += 1
                        continue

                    # Enumerate (v_a, v_o, v_q) triples that are
                    # pairwise distinct AND pairwise pose-diverse.
                    found_any = False
                    shuffled_a = list(only_a)
                    shuffled_o = list(only_o)
                    shuffled_q = list(only_q)
                    random.shuffle(shuffled_a)
                    random.shuffle(shuffled_o)
                    random.shuffle(shuffled_q)

                    for v_a in shuffled_a:
                        pose_a = pose_cache[v_a]
                        for v_o in shuffled_o:
                            if v_o == v_a:
                                continue
                            pose_o = pose_cache[v_o]
                            if not self._check_pose_diversity(
                                pose_o, [pose_a],
                                self.min_rot_angle, self.min_translation,
                            ):
                                continue
                            for v_q in shuffled_q:
                                if v_q == v_a or v_q == v_o:
                                    continue
                                pose_q = pose_cache[v_q]
                                if not self._check_pose_diversity(
                                    pose_q, [pose_a, pose_o],
                                    self.min_rot_angle, self.min_translation,
                                ):
                                    continue

                                candidates.append({
                                    "anchor_node": a_node,
                                    "orienting_node": o_node,
                                    "query_node": q_node,
                                    "v_a": int(v_a),
                                    "v_o": int(v_o),
                                    "v_q": int(v_q),
                                    "answer": label,
                                    "dx": dx,
                                    "dz": dz,
                                    "a_xy": (float(a_xy[0]), float(a_xy[1])),
                                    "o_xy": (float(o_xy[0]), float(o_xy[1])),
                                    "q_xy": (float(q_xy[0]), float(q_xy[1])),
                                })
                                found_any = True

                    if not found_any:
                        reject["no_pose_diverse_triple"] += 1

        return candidates, reject

    @staticmethod
    def _diverse_sample(candidates, n):
        """Round-robin diverse sample.

        Keyed first by anchor node, then by (anchor, orienting) pair,
        so one (anchor, orienting) pair does not monopolize the quota
        with many query-object / view-triple variants.
        """
        if n <= 0 or not candidates:
            return []
        if n >= len(candidates):
            out = list(candidates)
            random.shuffle(out)
            return out

        by_anchor = defaultdict(lambda: defaultdict(list))
        for c in candidates:
            anchor_id = c["anchor_node"].node_id
            pair_key = (
                str(c["anchor_node"].node_id),
                str(c["orienting_node"].node_id),
            )
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

    # ─── QA builder (single candidate) ──────────────────────────────────

    def _build_one_qa(self, graph, cand):
        a_node = cand["anchor_node"]
        o_node = cand["orienting_node"]
        q_node = cand["query_node"]
        v_a = cand["v_a"]
        v_o = cand["v_o"]
        v_q = cand["v_q"]
        answer_direction = cand["answer"]

        options = list(_DIRECTIONS_8)
        random.shuffle(options)
        answer_letter = _LETTERS[options.index(answer_direction)]
        options_str = "Options: " + " ".join(
            [f"{_LETTERS[i]}. {options[i]}" for i in range(len(options))]
        )
        question = (
            f"In image 1 you can see the {a_node.tag}. "
            f"In image 2 you can see the {o_node.tag}. "
            f"In image 3 you can see the {q_node.tag}. "
            f"If I stand at the {a_node.tag} and face the {o_node.tag}, "
            f"in which direction is the {q_node.tag}? " + options_str
        )
        prompt = question + " Answer: " + answer_letter

        # Images are passed in the order [anchor_view, orienting_view,
        # query_view] so they correspond 1-to-1 with "image 1/2/3" in
        # the prompt text.
        processed_images = [
            {"bytes": convert_pil_to_bytes(graph.views[v_a].image)},
            {"bytes": convert_pil_to_bytes(graph.views[v_o].image)},
            {"bytes": convert_pil_to_bytes(graph.views[v_q].image)},
        ]
        cog_ctx = self._make_cog_context(
            view_indices=[v_a, v_o, v_q],
            node_ids=[a_node.node_id, o_node.node_id, q_node.node_id],
            anchor_node_id=a_node.node_id,
        )
        # Reasoning overlay — carries the object-frame geometry so any
        # future renderer can draw the 8-sector wedge anchored at
        # ``a_node`` with forward aimed at ``o_node``. The stock BEV
        # renderer only consumes known kinds, so this block is a no-op
        # for visualization today; it is stored verbatim for audits /
        # dashboard tooling.
        if cog_ctx is not None:
            a_xy = cand["a_xy"]
            o_xy = cand["o_xy"]
            q_xy = cand["q_xy"]
            vf_x = o_xy[0] - a_xy[0]
            vf_y = o_xy[1] - a_xy[1]
            vf_n = math.hypot(vf_x, vf_y) or 1.0
            vf_x /= vf_n
            vf_y /= vf_n
            vr_x, vr_y = vf_y, -vf_x
            virtual_yaw_deg = float(math.degrees(math.atan2(vf_y, vf_x)))
            cog_ctx.extra["reasoning_overlay"] = {
                "kind": "mmsi_obj_face_obj_obj",
                # Per-role view indices — now three distinct views.
                "anchor_view_idx":    int(v_a),
                "orienting_view_idx": int(v_o),
                "query_view_idx":     int(v_q),
                # Back-compat single-view field (first image).
                "view_idx":           int(v_a),
                "dx": float(cand["dx"]),
                "dz": float(cand["dz"]),
                "answer": str(answer_direction),
                "anchor_world_xy": [a_xy[0], a_xy[1]],
                "orienting_world_xy": [o_xy[0], o_xy[1]],
                "query_world_xy": [q_xy[0], q_xy[1]],
                "anchor_tag": str(a_node.tag),
                "orienting_tag": str(o_node.tag),
                "query_tag": str(q_node.tag),
                # SceneNode.node_id is a *string* (multiview IDs are the
                # repr of the 9-dim float box). Never cast to int.
                "anchor_node_id":    str(a_node.node_id),
                "orienting_node_id": str(o_node.node_id),
                "query_node_id":     str(q_node.node_id),
                "virtual_fwd_xy":   [float(vf_x), float(vf_y)],
                "virtual_right_xy": [float(vr_x), float(vr_y)],
                "virtual_yaw_world_deg": virtual_yaw_deg,
            }
        return (prompt, processed_images, QuestionType.MCQ, cog_ctx,
                options, answer_letter)

    # ─── Handler ────────────────────────────────────────────────────────

    def _generate_object_facing_object_object_mcq(self, graph):
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
        distinct_anchors = len({c["anchor_node"].node_id for c in sampled})
        distinct_pairs = len({(str(c["anchor_node"].node_id),
                               str(c["orienting_node"].node_id))
                              for c in sampled})
        _p(
            f"scene={sid} sampled: quota={quota} pool={len(candidates)} "
            f"taken={len(sampled)} distinct_anchors={distinct_anchors} "
            f"distinct_pairs={distinct_pairs}"
        )

        results = []
        for k, cand in enumerate(sampled):
            (prompt, processed_images, qtype, cog_ctx,
             options, answer_letter) = self._build_one_qa(graph, cand)
            a_node = cand["anchor_node"]
            o_node = cand["orienting_node"]
            q_node = cand["query_node"]
            _p(
                f"scene={sid} qa[{k}]: views=({cand['v_a']},{cand['v_o']},"
                f"{cand['v_q']}) anchor={a_node.tag} orient={o_node.tag} "
                f"query={q_node.tag}  dx={cand['dx']:+.3f} "
                f"dz={cand['dz']:+.3f} -> {cand['answer']} "
                f"ans={answer_letter}  options={options}  "
                f"prompt={self._preview(prompt, 160)}"
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
