"""
MMSI-Bench: Camera–Camera direction task.

Given two diverse views that share at least one co-visible 3D object, asks
in which direction the second camera is located relative to the first.

Direction computation (horizontal-plane projection)
---------------------------------------------------
All data in this repo is z-up world (ARKitScenes / embodiedscan), so the
horizontal plane is world-xy. We do NOT work in A's raw 3D camera frame,
because hand-held cameras have non-negligible pitch/roll and the vertical
world-z component leaks into the camera's local (dx, dy, dz), polluting
|dx|/|dz| and yielding answers that disagree with what a human sees on the
BEV.

Instead, we:
  1. Take the world-xy offset ``Δ_xy = (B - A).xy``;
  2. Project A's camera forward/right basis vectors onto world-xy and
     renormalize to obtain ``fwd_hp``, ``right_hp`` (A's "heading" on the
     horizontal plane);
  3. Decompose ``Δ_xy = dx · right_hp + dz · fwd_hp``.
The resulting ``(dx, dz)`` is a pure horizontal-plane decomposition that
matches the BEV visualization exactly, regardless of camera tilt.

Answer space: 8 options — the octant set
{Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left}
permuted into A/B/C/D/E/F/G/H slots.

The bearing θ = atan2(dx, dz) is binned into eight 45°-wide octants
centered on 0°, ±45°, ±90°, ±135°, 180°. To keep the answer robust
against tiny pose noise, we reject pairs whose bearing lies within
``boundary_margin_deg`` of any octant boundary (22.5°, 67.5°, 112.5°,
157.5°). Default margin = 5°, i.e. the accepted safe zone is 35°
wide inside each 45° octant.

Frame-selection policy (B1: enumerate-then-sample)
--------------------------------------------------
For each scene we **enumerate** every (anchor_node, v_a, v_b) triple where:
  * ``anchor_node`` is visible in >= 2 views (co-visibility guarantee);
  * ``(v_a, v_b)`` is an unordered pair among those views;
  * both views carry a pose and pass the pose-diversity threshold
    (``min_rot_angle`` / ``min_translation``);
  * the horizontal-plane distance ``|Δ_xy| >= min_horizontal_translation``
    (hard floor — rejects near-stationary pairs where the direction is
    dominated by pose noise rather than a meaningful camera offset);
  * the vertical separation ``|Δ_z| <= max_vertical_translation``
    (hard ceiling — rejects pairs where one camera is crouched / the other
    is standing, since large height differences make the two images look
    like they were taken from different scene levels and corrupt the
    viewer's "front/back/left/right" intuition even though the BEV math
    itself is unaffected);
  * the direction is not degenerate (|dx|, |dz| not both < 5cm);
  * the bearing is inside an octant's safe zone, i.e. at least
    ``boundary_margin_deg`` away from any 45°-octant boundary.

Then we sample N = ``sub_tasks.camera_camera_mcq`` items from the pool with
a simple diversity rule (round-robin over distinct anchor_nodes, then over
distinct view-pairs) so the N QAs cover as many anchors / pairs as possible.

The handler is called **exactly once per scene** (``SUB_TASKS.default = 1``),
and returns a ``list[tuple]`` of up to N items. The generic dispatch in
``BaseAnnotationTask.process`` flattens the list automatically.

Diagnostic output
-----------------
All diagnostic lines are emitted via ``print(..., flush=True)`` with the
prefix ``[mmsi_cam_cam]`` so they survive ``tee`` / piping and can be
grepped out of the mixed pipeline log:

    grep "\\[mmsi_cam_cam\\]" logs/step5_demo_mmsi_camera_camera.log

Lines you will see per run:
  * init                 — thresholds the annotator runs with
  * scene=… entry        — view/pose/node counts on arrival
  * scene=… enum         — candidate-pool size + breakdown
  * scene=… sampled      — how many QAs we actually produce
  * scene=… qa[k]        — per-QA (anchor, v_a, v_b, dx, dz, answer, options)
  * scene=… status=OK|SKIP reason=…
  * summary              — at process exit: total / ok / skip + reason dist
"""

import atexit
import math
import random
from collections import Counter, defaultdict

import numpy as np

from .core.base_multiview_task import BaseMultiviewAnnotationTask
from .core.question_type import QuestionType
from utils.image_utils import convert_pil_to_bytes

_TAG = "[mmsi_cam_cam]"

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

# Default boundary-margin for the 8-way octant classifier. Offsets whose
# bearing lies within this many degrees of an octant boundary (22.5°,
# 67.5°, 112.5°, 157.5°, …) are rejected as ambiguous. 5° leaves a 35°
# "safe" inner zone inside each 45° octant.
_DEFAULT_BOUNDARY_MARGIN_DEG = 5.0

# Default hard floor on horizontal (world-xy) distance between the two
# cameras. Pairs closer than this on the horizontal plane are rejected
# outright: at sub-decimeter scales the "direction" is dominated by pose
# noise / hand-jitter rather than a meaningful viewpoint change.
_DEFAULT_MIN_HORIZONTAL_TRANSLATION = 0.3

# Default hard ceiling on vertical (world-z) distance between the two
# cameras. Pairs whose height difference exceeds this are rejected: large
# height gaps (e.g. one camera near the floor, the other near the ceiling)
# make the two images feel like they were captured at different scene
# levels, which disrupts the viewer's horizontal direction intuition even
# though the BEV-plane math itself is z-invariant. Keeping cameras roughly
# co-planar preserves the "two people walking around the same room"
# narrative the QA implicitly assumes.
_DEFAULT_MAX_VERTICAL_TRANSLATION = 0.5


def _p(msg):
    """Single print helper — keeps the prefix consistent and flushes."""
    print(f"{_TAG} {msg}", flush=True)


class AnnotationGenerator(BaseMultiviewAnnotationTask):

    QUESTION_TAG = "MMSI Camera-Camera"
    SUB_TASKS = {
        # ``default=1`` means: call handler once per scene. The handler
        # itself returns a list of up to ``count`` QA tuples, so setting
        # ``sub_tasks.camera_camera_mcq: N`` yields up to N QAs per scene.
        "camera_camera_mcq": {"default": 1, "handler": "_generate_camera_camera_mcq"},
    }

    # Process-wide counters aggregated across instances / threads.
    _SKIP_COUNTER = Counter()
    _SCENE_COUNTER = Counter()  # total / ok / skip
    _ATEXIT_REGISTERED = False

    def __init__(self, args):
        super().__init__(args)

        # How many QAs we want per scene (B1 quota). Falls back to 1.
        self._qa_quota = self.get_sub_task_count("camera_camera_mcq", default=1)

        # Octant-boundary margin (degrees) read from YAML.
        # Rejects pairs whose bearing sits within this many degrees of a
        # 45°-octant boundary (22.5° / 67.5° / 112.5° / 157.5° / …), to
        # avoid fence-sitting between e.g. Front and Front-Right. 0°
        # disables the filter.
        raw_margin = args.get("boundary_margin_deg",
                              _DEFAULT_BOUNDARY_MARGIN_DEG) \
            if hasattr(args, "get") \
            else getattr(args, "boundary_margin_deg",
                         _DEFAULT_BOUNDARY_MARGIN_DEG)
        try:
            self._boundary_margin_deg = max(0.0, float(raw_margin))
        except (TypeError, ValueError):
            self._boundary_margin_deg = _DEFAULT_BOUNDARY_MARGIN_DEG
        # Cap at just under half an octant (22.5°); any larger collapses
        # the accepted region to nothing.
        self._boundary_margin_deg = min(self._boundary_margin_deg, 22.4)

        # Horizontal-plane translation hard floor. Unlike ``min_translation``
        # (which is an OR-combined 3D pose-diversity metric shared with other
        # tasks), this one is an AND-combined hard filter applied only to
        # camera-camera direction pairs. Rationale: direction judgements at
        # <0.3 m horizontal separation are dominated by noise even when the
        # cameras rotated a lot.
        raw_hmin = args.get("min_horizontal_translation",
                            _DEFAULT_MIN_HORIZONTAL_TRANSLATION) \
            if hasattr(args, "get") \
            else getattr(args, "min_horizontal_translation",
                         _DEFAULT_MIN_HORIZONTAL_TRANSLATION)
        try:
            self._min_horizontal_translation = max(0.0, float(raw_hmin))
        except (TypeError, ValueError):
            self._min_horizontal_translation = _DEFAULT_MIN_HORIZONTAL_TRANSLATION

        # Vertical (world-z) translation hard ceiling. BEV math is already
        # z-invariant, but large height gaps between A and B visually skew
        # the viewer's direction perception (one image looks like it was
        # taken at a different scene level). Reject pairs with |Δz| above
        # this threshold. Set to a large number / inf to disable.
        raw_vmax = args.get("max_vertical_translation",
                            _DEFAULT_MAX_VERTICAL_TRANSLATION) \
            if hasattr(args, "get") \
            else getattr(args, "max_vertical_translation",
                         _DEFAULT_MAX_VERTICAL_TRANSLATION)
        try:
            self._max_vertical_translation = max(0.0, float(raw_vmax))
        except (TypeError, ValueError):
            self._max_vertical_translation = _DEFAULT_MAX_VERTICAL_TRANSLATION

        _p(
            f"init: min_rot={self.min_rot_angle:.1f}°  "
            f"min_trans={self.min_translation:.2f}m  "
            f"min_hp_trans={self._min_horizontal_translation:.2f}m  "
            f"max_vert_trans={self._max_vertical_translation:.2f}m  "
            f"boundary_margin={self._boundary_margin_deg:.1f}°  "
            f"max_num_views={self.max_num_views}  "
            f"quota_per_scene={self._qa_quota}  "
            f"policy=enumerate_then_sample(num_views=2) answer_space=8way"
        )

        if not AnnotationGenerator._ATEXIT_REGISTERED:
            atexit.register(AnnotationGenerator._dump_summary)
            AnnotationGenerator._ATEXIT_REGISTERED = True

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
    def _pose_brief(pose):
        """Compact 1-line summary of a 4x4 pose: translation + rot-from-I angle."""
        if pose is None:
            return "None"
        t = pose[:3, 3]
        R = pose[:3, :3]
        trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(trace)))
        return (f"t=({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f}) "
                f"|rot_from_I|={ang:.1f}°")

    @staticmethod
    def _preview(text, n=90):
        if not isinstance(text, str):
            return str(text)
        text = text.replace("\n", " ").strip()
        return text if len(text) <= n else text[:n - 1] + "…"

    def _print_entry_snapshot(self, graph):
        sid = self._scene_id(graph)
        view_ids = sorted(graph.views.keys())
        view_ids_w_pose = [v for v in view_ids
                           if graph.views[v].pose is not None]
        first_img = (graph.views[view_ids[0]].image_path
                     if view_ids else "(no views)")
        _p(
            f"scene={sid} entry: views={len(view_ids)} "
            f"(w/pose={len(view_ids_w_pose)}) "
            f"nodes={len(graph.nodes)} "
            f"box_to_view={len(getattr(graph, 'box_to_view_proj', {}) or {})}  "
            f"view_idx_range=[{view_ids[0] if view_ids else '-'}.."
            f"{view_ids[-1] if view_ids else '-'}]  "
            f"first_image={first_img}"
        )

    # ─── Direction classification ────────────────────────────────────────

    @staticmethod
    def _classify_direction(pose_a, pose_b):
        """Classify B's direction relative to A on the horizontal plane.

        Returns ``(answer_direction, dx, dz)`` or ``(None, dx, dz)`` when
        the horizontal offset is degenerate (both components <5cm).

        The decomposition is performed entirely on the world horizontal
        plane (world-xy, with world-z = up) to immunize the answer against
        camera pitch/roll. Concretely:

          Δ_xy   = (B - A).xy                                        (2-vec)
          fwd_hp = normalize( (R_A @ e_z).xy )    # A's forward on xy
          right_hp = normalize( (R_A @ e_x).xy )  # A's right   on xy
          dz = Δ_xy · fwd_hp        (+ = in front of A)
          dx = Δ_xy · right_hp      (+ = on A's right)

        Semantics of (dx, dz) are identical to the old camera-frame version
        (+z forward, +x right), but computed in a way that is consistent
        with the BEV rendering (which also uses fwd_xy / right_xy from
        ``pose[:3,:3] @ e_{z,x}`` and projects onto world-xy).

        Degeneracy fallback: if A's forward/right have no xy component
        (camera looking straight up or down), we fall back to the old
        camera-frame formula so the classifier never crashes.

        Note: this classifier does NOT enforce that the bearing is well
        clear of an octant boundary — that's the job of
        `_is_octant_unambiguous`, applied at enumeration time. The
        classifier only labels; the filter decides admissibility.
        """
        pose_a = np.asarray(pose_a, dtype=float)
        pose_b = np.asarray(pose_b, dtype=float)

        # Horizontal offset in world-xy.
        dxy = pose_b[:3, 3] - pose_a[:3, 3]
        dxw, dyw = float(dxy[0]), float(dxy[1])

        # A's forward / right basis vectors projected onto world-xy.
        R_a = pose_a[:3, :3]
        fwd_w = R_a @ np.array([0.0, 0.0, 1.0])   # camera +Z in world
        right_w = R_a @ np.array([1.0, 0.0, 0.0])  # camera +X in world
        fwd_xy = np.array([float(fwd_w[0]), float(fwd_w[1])])
        right_xy = np.array([float(right_w[0]), float(right_w[1])])

        fwd_n = float(np.linalg.norm(fwd_xy))
        right_n = float(np.linalg.norm(right_xy))

        if fwd_n < 1e-6 or right_n < 1e-6:
            # Camera aimed (nearly) straight up/down — horizontal basis
            # is undefined. Fall back to the camera-frame formula so we
            # still return *something* deterministic.
            cam_b_in_a = np.linalg.inv(pose_a) @ np.array([*pose_b[:3, 3], 1.0])
            x = float(cam_b_in_a[0])
            z = float(cam_b_in_a[2])
        else:
            fwd_xy /= fwd_n
            right_xy /= right_n
            x = dxw * float(right_xy[0]) + dyw * float(right_xy[1])
            z = dxw * float(fwd_xy[0]) + dyw * float(fwd_xy[1])

        if abs(x) < 0.05 and abs(z) < 0.05:
            return None, x, z
        # 8-way classification: bin the bearing θ = atan2(dx, dz) into
        # 45°-wide octants centered on 0°, ±45°, ±90°, ±135°, 180°.
        # Mapping (θ in degrees, wrapped to [-180, 180)):
        #   [-22.5,  22.5) → Front
        #   [ 22.5,  67.5) → Front-Right
        #   [ 67.5, 112.5) → Right
        #   [112.5, 157.5) → Back-Right
        #   [157.5, 180) ∪ [-180, -157.5) → Back
        #   [-157.5, -112.5) → Back-Left
        #   [-112.5,  -67.5) → Left
        #   [-67.5,   -22.5) → Front-Left
        bearing_deg = math.degrees(math.atan2(x, z))  # -180..180, 0=Front
        # Shift so that the first octant (Front) starts at 0° and each
        # octant spans 45°.
        idx = int(((bearing_deg + 22.5) % 360.0) // 45.0)
        direction = _DIRECTIONS_8[idx]
        return direction, x, z

    @staticmethod
    def _is_octant_unambiguous(dx, dz, margin_deg):
        """True iff the (dx, dz) bearing is inside an octant's safe zone.

        Each of the 8 octants covers 45°. We reject pairs whose bearing
        lies within ``margin_deg`` of any boundary (±22.5°, ±67.5°, etc.),
        so the accepted zone inside each octant is (45 − 2·margin)° wide.
        When ``margin_deg <= 0`` this degenerates to True.
        """
        if margin_deg <= 0.0:
            return True
        ax, az = abs(dx), abs(dz)
        if ax < 1e-6 and az < 1e-6:
            return True  # caught earlier as degenerate; be safe
        bearing_deg = math.degrees(math.atan2(dx, dz))
        # Distance (in degrees) from the nearest octant center (0, ±45,
        # ±90, ±135, 180). Centers occur every 45°; the signed offset
        # from the nearest center is ((θ + 22.5) mod 45) − 22.5.
        offset = ((bearing_deg + 22.5) % 45.0) - 22.5
        # Inside the octant iff |offset| <= (22.5 − margin).
        return abs(offset) <= (22.5 - margin_deg)

    # ─── Candidate enumeration & diverse sampling ────────────────────────

    def _enumerate_cam_cam_candidates(self, graph):
        """Enumerate every legal (anchor_node, v_a, v_b) triple.

        Filters applied:
          * anchor_node visible in >= 2 views (co-visibility);
          * both v_a, v_b have poses;
          * pose-diversity threshold (min_rot_angle / min_translation);
          * non-degenerate direction (|dx|, |dz| not both < 5cm).

        Returns:
            list[dict] — each dict has: node, v_a, v_b, pose_a, pose_b,
                        answer, dx, dz.
            Plus a ``reject`` Counter recording why triples were dropped.
        """
        candidates = []
        reject = Counter()

        box_to_views = getattr(graph, "box_to_view_proj", {}) or {}
        for nid, view_list in box_to_views.items():
            node = graph.nodes.get(nid)
            if node is None:
                continue
            # Only views that actually have an appearance on this node.
            views = [vi for vi in view_list if vi in node.view_appearances]
            # Require both views to have poses.
            views = [vi for vi in views if graph.views[vi].pose is not None]
            if len(views) < 2:
                reject["anchor_under_2_posed_views"] += 1
                continue

            # Unordered pairs. Note (v_a, v_b) and (v_b, v_a) give different
            # questions ("B relative to A" ≠ "A relative to B"), but the
            # direction class flips trivially. We keep it unordered so the
            # pool stays clean; the image order is assigned at QA build time.
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    v_a, v_b = views[i], views[j]
                    pose_a = np.asarray(graph.views[v_a].pose, dtype=float)
                    pose_b = np.asarray(graph.views[v_b].pose, dtype=float)

                    # Pose diversity: use the same check as _find_overlapping_views,
                    # but pairwise (selected_poses=[pose_a]).
                    if not self._check_pose_diversity(
                        pose_b, [pose_a],
                        self.min_rot_angle, self.min_translation,
                    ):
                        reject["pose_not_diverse"] += 1
                        continue

                    # Horizontal-plane translation hard floor. The 3D
                    # pose-diversity check above is OR-combined with
                    # rotation, so it can let through pairs that differ
                    # mostly in yaw but barely moved on the horizontal
                    # plane — exactly the case that makes the direction
                    # answer noise-dominated. Require a real horizontal
                    # separation before the pair is admitted.
                    if self._min_horizontal_translation > 0.0:
                        dxy_hp = pose_b[:3, 3] - pose_a[:3, 3]
                        hp_dist = float(math.hypot(float(dxy_hp[0]),
                                                   float(dxy_hp[1])))
                        if hp_dist < self._min_horizontal_translation:
                            reject["horizontal_too_close"] += 1
                            continue

                    # Vertical translation hard ceiling. The BEV-plane
                    # decomposition that produces (dx, dz) is z-invariant
                    # by construction, so large |Δz| does NOT break the
                    # numeric answer. However it visibly breaks the
                    # viewer's direction intuition: e.g. A at 1.7m vs B
                    # at 0.3m looks like two totally different vantage
                    # levels, and humans rating those two frames will
                    # struggle to map "front/back/left/right" onto what
                    # they see. Reject such pairs so the dataset stays
                    # in the "both cameras at roughly the same height"
                    # regime that the question assumes.
                    if self._max_vertical_translation > 0.0:
                        dz_world = float(pose_b[2, 3] - pose_a[2, 3])
                        if abs(dz_world) > self._max_vertical_translation:
                            reject["vertical_too_far"] += 1
                            continue

                    answer, dx, dz = self._classify_direction(pose_a, pose_b)
                    if answer is None:
                        reject["degenerate_direction"] += 1
                        continue

                    # Reject offsets that sit too close to an octant
                    # boundary (22.5° / 67.5° / 112.5° / 157.5° …). This
                    # is the 8-way analogue of the old cardinal-dominance
                    # filter: it keeps answers safely inside one octant
                    # so tiny pose noise cannot flip the label to a
                    # neighbouring octant.
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
                        "answer": answer,
                        "dx": dx,
                        "dz": dz,
                    })
        return candidates, reject

    @staticmethod
    def _diverse_sample(candidates, n):
        """Sample up to n items with round-robin diversity.

        Strategy (two-level RR):
          1. Group candidates by ``node.node_id`` (anchor).
          2. Within each anchor, group by unordered pair ``(min(v_a,v_b), max(...))``.
          3. Round-robin: at each step pick one anchor (rotating), then
             within that anchor pick one view-pair (rotating), then pop one
             candidate randomly from that pair bucket.

        This spreads picks across anchors first, then across distinct view
        pairs, while still leaving room for duplicate-answer diversity within
        a pair (rare — same pair usually gives the same direction).
        """
        if n <= 0 or not candidates:
            return []
        if n >= len(candidates):
            # Small pool: return all, shuffled for stable randomness.
            out = list(candidates)
            random.shuffle(out)
            return out

        # Group by anchor → pair → [candidates]
        by_anchor = defaultdict(lambda: defaultdict(list))
        for c in candidates:
            anchor_id = c["node"].node_id
            pair_key = (min(c["v_a"], c["v_b"]), max(c["v_a"], c["v_b"]))
            by_anchor[anchor_id][pair_key].append(c)

        anchor_ids = list(by_anchor.keys())
        random.shuffle(anchor_ids)
        # Shuffle inner buckets too.
        for aid in anchor_ids:
            pair_keys = list(by_anchor[aid].keys())
            random.shuffle(pair_keys)
            by_anchor[aid] = {pk: by_anchor[aid][pk] for pk in pair_keys}
            for pk in by_anchor[aid]:
                random.shuffle(by_anchor[aid][pk])

        picked = []
        # Round-robin across anchors; within each anchor, round-robin over
        # pair-buckets; pop one candidate per visit.
        while len(picked) < n:
            progressed = False
            for aid in list(anchor_ids):
                buckets = by_anchor[aid]
                if not buckets:
                    anchor_ids.remove(aid)
                    continue
                # Pick the first (already-shuffled) non-empty pair bucket.
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

    # ─── QA builder (single pair) ────────────────────────────────────────

    def _build_one_qa(self, graph, cand):
        """Turn a single candidate dict into a QA 4-tuple."""
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
            "In image 2, the camera is located in which direction relative to "
            "image 1's camera? " + options_str
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
        # Reasoning overlay — consumed by CognitiveMapRenderer to draw A's
        # local axes, the A→B connector, and the dx/dz decomposition on
        # top of the BEV. Safe to attach even if cog map rendering is
        # disabled (builder silently skips when the map is not built).
        if cog_ctx is not None:
            # World-frame horizontal-plane (xy) positions of both cameras
            # — handy for the reasoning panel that shows "two cameras'
            # absolute positions + their Δxy" alongside the A-local
            # (dx, dz) decomposition that actually drives the answer.
            # Convention in this repo: z-up world, so xy is the horizontal
            # plane (ARKitScenes / embodiedscan). yaw_world is the angle
            # of A's forward vector projected onto world-xy, measured
            # counter-clockwise from world +x.
            pose_a = cand["pose_a"]
            pose_b = cand["pose_b"]
            a_wxy = (float(pose_a[0, 3]), float(pose_a[1, 3]))
            b_wxy = (float(pose_b[0, 3]), float(pose_b[1, 3]))
            fwd_a_world = pose_a[:3, :3] @ np.array([0.0, 0.0, 1.0])
            a_yaw_world_deg = float(math.degrees(
                math.atan2(float(fwd_a_world[1]), float(fwd_a_world[0]))
            ))
            cog_ctx.extra["reasoning_overlay"] = {
                "kind": "mmsi_cam_cam",
                "anchor_view_idx": int(v_a),
                "target_view_idx": int(v_b),
                "dx": float(cand["dx"]),
                "dz": float(cand["dz"]),
                "answer": str(answer_direction),
                # World-frame horizontal-plane (xy) info
                "a_world_xy": [a_wxy[0], a_wxy[1]],
                "b_world_xy": [b_wxy[0], b_wxy[1]],
                "delta_world_xy": [b_wxy[0] - a_wxy[0], b_wxy[1] - a_wxy[1]],
                "a_yaw_world_deg": a_yaw_world_deg,
            }
        return prompt, processed_images, QuestionType.MCQ, cog_ctx, options, answer_letter

    # ─── Handler ─────────────────────────────────────────────────────────

    def _generate_camera_camera_mcq(self, graph):
        """B1 scheme: enumerate all legal pairs, then diversely sample N."""
        AnnotationGenerator._SCENE_COUNTER["total"] += 1
        sid = self._scene_id(graph)

        # Layer 1 — entry snapshot.
        self._print_entry_snapshot(graph)

        n_views = len(graph.views)
        n_poses = sum(1 for vi in graph.views
                      if graph.views[vi].pose is not None)
        n_nodes = len(graph.nodes)

        # Layer 2 — enumerate candidate pool.
        candidates, reject = self._enumerate_cam_cam_candidates(graph)
        _p(
            f"scene={sid} enum: candidates={len(candidates)}  "
            f"rejects={dict(reject) if reject else '{}'}"
        )

        if not candidates:
            reason = "no_candidates"
            AnnotationGenerator._record_skip(reason)
            AnnotationGenerator._SCENE_COUNTER["skip"] += 1
            _p(
                f"scene={sid} views={n_views} poses={n_poses} nodes={n_nodes} "
                f"status=SKIP reason={reason}"
            )
            return None

        # Layer 3 — diverse sampling up to quota.
        quota = max(int(self._qa_quota), 1)
        sampled = self._diverse_sample(candidates, quota)
        # Count distinct anchors / pairs actually used — purely for logging.
        distinct_anchors = len({c["node"].node_id for c in sampled})
        distinct_pairs = len({(min(c["v_a"], c["v_b"]),
                                max(c["v_a"], c["v_b"])) for c in sampled})
        _p(
            f"scene={sid} sampled: quota={quota} pool={len(candidates)} "
            f"taken={len(sampled)} distinct_anchors={distinct_anchors} "
            f"distinct_pairs={distinct_pairs}"
        )

        # Layer 4 — build QAs.
        results = []
        for k, cand in enumerate(sampled):
            prompt, processed_images, qtype, cog_ctx, options, answer_letter = \
                self._build_one_qa(graph, cand)
            v_a, v_b = cand["v_a"], cand["v_b"]
            node = cand["node"]
            _p(
                f"scene={sid} qa[{k}]: anchor={node.tag}({node.node_id}) "
                f"pair=({v_a},{v_b}) dx={cand['dx']:+.3f} dz={cand['dz']:+.3f} "
                f"-> {cand['answer']} ans={answer_letter}  "
                f"options={options}  "
                f"prompt={self._preview(prompt, 120)}"
            )
            results.append((prompt, processed_images, qtype, cog_ctx))

        if not results:
            reason = "sample_empty"
            AnnotationGenerator._record_skip(reason)
            AnnotationGenerator._SCENE_COUNTER["skip"] += 1
            _p(
                f"scene={sid} views={n_views} poses={n_poses} nodes={n_nodes} "
                f"status=SKIP reason={reason}"
            )
            return None

        AnnotationGenerator._SCENE_COUNTER["ok"] += 1
        AnnotationGenerator._SCENE_COUNTER["qa_ok"] += len(results)
        _p(
            f"scene={sid} views={n_views} poses={n_poses} nodes={n_nodes} "
            f"status=OK qa_generated={len(results)} pool={len(candidates)}"
        )
        return results
