"""
Cognitive Map core module for OpenSpatial annotation tasks.

Provides:
- `CognitiveMapContext`: runtime metadata describing which views/nodes a QA uses.
- `CognitiveMapBuilder`: builds a structured 10x10 grid cognitive map dict from
  a SceneGraph and a context (camera pose + object 3D boxes projected to xy plane).
- `CognitiveMapRenderer`: renders the map (+ question/answer) into a PNG bytes.
- `generate_bev_perturbations`: creates distractor BEV layouts for MCQ option images.

The builder is completely side-effect free and safe to call in multi-worker
pipelines. The renderer uses matplotlib's non-interactive ``Agg`` backend.
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Context ──────────────────────────────────────────────────────────────

@dataclass
class CognitiveMapContext:
    """Context describing which views/nodes participate in a single QA.

    Attributes:
        view_indices: view ids used in this QA (camera viewpoints to plot).
        node_ids: object node ids referenced in the QA (objects to plot).
        anchor_node_id: optional node id to highlight as "anchor" (different color).
        extra: free-form dict for task-specific metadata (not persisted by default).
    """
    view_indices: List[int] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    anchor_node_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.view_indices and not self.node_ids


# ─── Builder ──────────────────────────────────────────────────────────────

class CognitiveMapBuilder:
    """Builds a structured cognitive map dict from a SceneGraph + context.

    Output schema (all types JSON-serializable):
        {
            "grid_size": [10, 10],
            "bounds": [xmin, ymin, xmax, ymax],      # xy world range (10% padded)
            "cells": 10x10 int matrix,               # occupancy counts per cell
            "cameras": [ {view_idx, position_xy, yaw_deg, cell_idx}, ... ],
            "objects": [ {tag, node_id, center_xy, size_xy, yaw_deg, cell_idx,
                          is_anchor}, ... ],
        }
    """

    def __init__(self, grid_size: int = 10, padding_ratio: float = 0.10,
                 default_extent: float = 1.0):
        """
        Args:
            grid_size: number of cells per axis (default 10 -> 10x10 grid).
            padding_ratio: expansion ratio applied to xy bounds (0.10 = +10%).
            default_extent: min extent (meters) used when all entities collapse
                            to a single point, to avoid a zero-size bbox.
        """
        self.grid_size = grid_size
        self.padding_ratio = padding_ratio
        self.default_extent = default_extent

    # ── public ────────────────────────────────────────────────────────

    def build(self, graph, context: CognitiveMapContext) -> Optional[Dict[str, Any]]:
        """Build a cognitive map dict. Returns None if no usable entity exists."""
        if context is None or context.is_empty():
            return None

        cameras_raw = self._collect_cameras(graph, context.view_indices)
        objects_raw = self._collect_objects(graph, context.node_ids,
                                            context.anchor_node_id)

        if not cameras_raw and not objects_raw:
            return None

        bounds = self._compute_bounds(cameras_raw, objects_raw)
        xmin, ymin, xmax, ymax = bounds

        cells = [[0 for _ in range(self.grid_size)]
                 for _ in range(self.grid_size)]

        cameras_out: List[Dict[str, Any]] = []
        for cam in cameras_raw:
            row, col = self._xy_to_cell(cam["position_xy"], bounds)
            cells[row][col] += 1
            cameras_out.append({
                "view_idx": cam["view_idx"],
                "position_xy": [float(cam["position_xy"][0]),
                                float(cam["position_xy"][1])],
                "yaw_deg": float(cam["yaw_deg"]),
                "cell_idx": [row, col],
            })

        objects_out: List[Dict[str, Any]] = []
        for obj in objects_raw:
            row, col = self._xy_to_cell(obj["center_xy"], bounds)
            cells[row][col] += 1
            objects_out.append({
                "tag": obj["tag"],
                "node_id": obj["node_id"],
                "center_xy": [float(obj["center_xy"][0]),
                              float(obj["center_xy"][1])],
                "size_xy": [float(obj["size_xy"][0]),
                            float(obj["size_xy"][1])],
                "yaw_deg": float(obj["yaw_deg"]),
                "cell_idx": [row, col],
                "is_anchor": bool(obj["is_anchor"]),
            })

        return {
            "grid_size": [self.grid_size, self.grid_size],
            "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "cells": cells,
            "cameras": cameras_out,
            "objects": objects_out,
        }

    # ── internals ─────────────────────────────────────────────────────

    def _collect_cameras(self, graph, view_indices: List[int]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for vi in view_indices:
            if vi in seen:
                continue
            seen.add(vi)
            view = graph.views.get(vi) if hasattr(graph, "views") else None
            if view is None:
                continue
            pose = None
            try:
                pose = view.pose  # cached property; may raise if missing
            except Exception:
                pose = None
            if pose is None:
                continue
            pose = np.asarray(pose)
            if pose.shape != (4, 4):
                continue
            # Camera center in world (pose is camera-to-world).
            pos_xy = pose[:3, 3][:2]
            # Forward vector in camera frame is +Z (pinhole convention used in
            # this repo; see utils.box_utils.convert_box_3d_world_to_camera).
            # We rotate (0,0,1) to world: forward_world = R @ [0,0,1] = R[:,2].
            forward_world = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
            yaw_rad = math.atan2(forward_world[1], forward_world[0])
            out.append({
                "view_idx": int(vi),
                "position_xy": np.array([pos_xy[0], pos_xy[1]], dtype=float),
                "yaw_deg": math.degrees(yaw_rad),
            })
        return out

    def _collect_objects(self, graph, node_ids: List[str],
                         anchor_node_id: Optional[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for nid in node_ids:
            if nid is None or nid in seen:
                continue
            seen.add(nid)
            node = graph.nodes.get(nid) if hasattr(graph, "nodes") else None
            if node is None:
                continue
            box = node.box_3d_world
            if box is None or len(box) < 6:
                continue
            cx, cy = float(box[0]), float(box[1])
            xl, yl = float(box[3]), float(box[4])
            yaw_deg = float(math.degrees(box[8])) if len(box) >= 9 else 0.0
            out.append({
                "tag": node.tag,
                "node_id": str(nid),
                "center_xy": np.array([cx, cy], dtype=float),
                "size_xy": np.array([xl, yl], dtype=float),
                "yaw_deg": yaw_deg,
                "is_anchor": (anchor_node_id is not None and nid == anchor_node_id),
            })
        return out

    def _compute_bounds(self, cameras, objects) -> Tuple[float, float, float, float]:
        xs: List[float] = []
        ys: List[float] = []
        for c in cameras:
            xs.append(float(c["position_xy"][0]))
            ys.append(float(c["position_xy"][1]))
        for o in objects:
            cx = float(o["center_xy"][0])
            cy = float(o["center_xy"][1])
            sx = float(o["size_xy"][0]) / 2.0
            sy = float(o["size_xy"][1]) / 2.0
            xs.extend([cx - sx, cx + sx])
            ys.extend([cy - sy, cy + sy])

        if not xs or not ys:
            # Defensive: caller should have filtered this case already.
            return (-self.default_extent, -self.default_extent,
                    self.default_extent, self.default_extent)

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Guarantee non-zero extent.
        if xmax - xmin < 1e-6:
            xmin -= self.default_extent / 2.0
            xmax += self.default_extent / 2.0
        if ymax - ymin < 1e-6:
            ymin -= self.default_extent / 2.0
            ymax += self.default_extent / 2.0

        dx = xmax - xmin
        dy = ymax - ymin
        xmin -= dx * self.padding_ratio
        xmax += dx * self.padding_ratio
        ymin -= dy * self.padding_ratio
        ymax += dy * self.padding_ratio
        return xmin, ymin, xmax, ymax

    def _xy_to_cell(self, pos_xy, bounds) -> Tuple[int, int]:
        xmin, ymin, xmax, ymax = bounds
        g = self.grid_size
        col = int((float(pos_xy[0]) - xmin) / (xmax - xmin) * g)
        row = int((float(pos_xy[1]) - ymin) / (ymax - ymin) * g)
        col = max(0, min(g - 1, col))
        row = max(0, min(g - 1, row))
        return row, col


# ─── Renderer ─────────────────────────────────────────────────────────────

def _get_matplotlib():
    """Lazy import matplotlib with a non-interactive (thread-safe) backend."""
    import matplotlib
    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        # If already set to another backend, keep it; Agg is only a preference.
        pass
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    return matplotlib, plt, Rectangle


class CognitiveMapRenderer:
    """Render a cognitive map dict + QA text into a PNG bytes image."""

    # Visual constants.
    CAMERA_COLOR = "#1f77b4"       # blue
    OBJECT_COLOR = "#2ca02c"       # green
    ANCHOR_COLOR = "#d62728"       # red
    GRID_COLOR = "#bfbfbf"
    BACKGROUND = "#ffffff"

    def __init__(self, figsize=(6.5, 7.5), dpi=120, arrow_length_ratio=0.08,
                 max_question_lines: int = 3, max_chars_per_line: int = 80):
        self.figsize = figsize
        self.dpi = dpi
        self.arrow_length_ratio = arrow_length_ratio
        self.max_question_lines = max_question_lines
        self.max_chars_per_line = max_chars_per_line

    # ── public ────────────────────────────────────────────────────────

    def render(self, cognitive_map: Dict[str, Any], question: str = "",
               answer: str = "") -> Optional[bytes]:
        """Render the cognitive map. Returns PNG bytes or None on failure."""
        if cognitive_map is None:
            return None
        try:
            return self._render_impl(cognitive_map, question, answer)
        except Exception:
            # Never let rendering errors break the pipeline.
            return None

    def render_bev_only(self, cognitive_map: Dict[str, Any],
                        title: str = "") -> Optional[bytes]:
        """Render the BEV part only (used for MCQ option diagrams)."""
        if cognitive_map is None:
            return None
        try:
            return self._render_impl(cognitive_map, question="", answer="",
                                     bev_only=True, title=title)
        except Exception:
            return None

    # ── internals ─────────────────────────────────────────────────────

    def _render_impl(self, cmap: Dict[str, Any], question: str, answer: str,
                     *, bev_only: bool = False, title: str = "") -> bytes:
        _, plt, Rectangle = _get_matplotlib()

        bounds = cmap.get("bounds")
        if bounds is None or len(bounds) != 4:
            return None  # type: ignore[return-value]
        xmin, ymin, xmax, ymax = bounds
        grid_size = cmap.get("grid_size", [10, 10])
        gx, gy = int(grid_size[0]), int(grid_size[1])

        if bev_only:
            fig = plt.figure(figsize=(self.figsize[0], self.figsize[0]),
                             dpi=self.dpi)
            ax_bev = fig.add_subplot(1, 1, 1)
            ax_text = None
        else:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.0], hspace=0.25)
            ax_bev = fig.add_subplot(gs[0, 0])
            ax_text = fig.add_subplot(gs[1, 0])

        fig.patch.set_facecolor(self.BACKGROUND)

        # Axis setup.
        ax_bev.set_xlim(xmin, xmax)
        ax_bev.set_ylim(ymin, ymax)
        ax_bev.set_aspect("equal", adjustable="box")
        ax_bev.set_xlabel("world X (m)")
        ax_bev.set_ylabel("world Y (m)")
        if title:
            ax_bev.set_title(title, fontsize=10)

        # Draw 10x10 grid lines.
        for i in range(gx + 1):
            x = xmin + (xmax - xmin) * i / gx
            ax_bev.plot([x, x], [ymin, ymax], color=self.GRID_COLOR,
                        linewidth=0.6, alpha=0.6, zorder=1)
        for j in range(gy + 1):
            y = ymin + (ymax - ymin) * j / gy
            ax_bev.plot([xmin, xmax], [y, y], color=self.GRID_COLOR,
                        linewidth=0.6, alpha=0.6, zorder=1)

        # Draw objects (AABB).
        extent = max(xmax - xmin, ymax - ymin)
        for obj in cmap.get("objects", []):
            cx, cy = obj["center_xy"]
            sx, sy = obj["size_xy"]
            sx = max(float(sx), extent * 0.01)
            sy = max(float(sy), extent * 0.01)
            color = self.ANCHOR_COLOR if obj.get("is_anchor") else self.OBJECT_COLOR
            rect = Rectangle((cx - sx / 2.0, cy - sy / 2.0), sx, sy,
                             fill=True, facecolor=color, alpha=0.25,
                             edgecolor=color, linewidth=1.2, zorder=3)
            ax_bev.add_patch(rect)
            ax_bev.plot(cx, cy, marker="s", color=color, markersize=4, zorder=4)
            ax_bev.text(cx, cy + sy / 2.0 + extent * 0.015,
                        str(obj.get("tag", "")),
                        fontsize=8, ha="center", va="bottom", color=color,
                        zorder=5)

        # Draw cameras (dot + arrow + label).
        arrow_len = extent * self.arrow_length_ratio
        for cam in cmap.get("cameras", []):
            cx, cy = cam["position_xy"]
            yaw_deg = float(cam.get("yaw_deg", 0.0))
            dx = arrow_len * math.cos(math.radians(yaw_deg))
            dy = arrow_len * math.sin(math.radians(yaw_deg))
            ax_bev.plot(cx, cy, marker="o", color=self.CAMERA_COLOR,
                        markersize=6, zorder=5)
            ax_bev.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color=self.CAMERA_COLOR,
                                linewidth=1.6),
                zorder=6,
            )
            ax_bev.text(cx + extent * 0.012, cy + extent * 0.012,
                        f"View {cam.get('view_idx', '?')}",
                        fontsize=8, color=self.CAMERA_COLOR, zorder=6)

        # Question / answer text panel.
        if ax_text is not None:
            ax_text.axis("off")
            wrapped_q = self._wrap_text(question or "")
            q_display = "\n".join(wrapped_q[: self.max_question_lines]) \
                if wrapped_q else ""
            ax_text.text(0.0, 0.95, f"Q: {q_display}",
                         fontsize=9, ha="left", va="top",
                         transform=ax_text.transAxes, wrap=True)
            ax_text.text(0.0, 0.25, f"A: {answer or ''}",
                         fontsize=10, ha="left", va="top",
                         transform=ax_text.transAxes,
                         color="#b8860b", fontweight="bold", wrap=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _wrap_text(self, text: str) -> List[str]:
        if not text:
            return []
        # Simple greedy word wrap. Matplotlib's own wrap is unreliable across
        # layout managers, so we pre-wrap explicitly.
        words = text.split()
        lines: List[str] = []
        current = ""
        for w in words:
            candidate = (current + " " + w).strip()
            if len(candidate) > self.max_chars_per_line and current:
                lines.append(current)
                current = w
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines


# ─── BEV perturbation (for Camera Pose Estimation MCQ distractors) ────────

def generate_bev_perturbations(cognitive_map: Dict[str, Any],
                               n: int = 3,
                               rng: Optional[random.Random] = None
                               ) -> List[Dict[str, Any]]:
    """Generate ``n`` distractor BEV layouts from a true cognitive map.

    Each distractor is a deep-ish copy with one of the following perturbations
    applied to its *cameras* only (objects are left intact for visual anchoring):

    - "mirror_x": flip camera x-coordinates and yaw about the scene center.
    - "mirror_y": flip camera y-coordinates and yaw about the scene center.
    - "rotate_180": rotate all camera positions and yaw by 180 degrees around
      the scene center.
    - "shuffle": permute the camera positions (but keep yaws), creating a
      plausible-but-wrong spatial configuration.

    The function picks ``n`` distinct perturbations; if fewer than ``n`` unique
    perturbations are available the list is padded with repeats.
    """
    if cognitive_map is None:
        return []
    if rng is None:
        rng = random.Random()

    ops = ["mirror_x", "mirror_y", "rotate_180", "shuffle"]
    rng.shuffle(ops)
    chosen = ops[:n] if len(ops) >= n else ops + [ops[0]] * (n - len(ops))

    out: List[Dict[str, Any]] = []
    for op in chosen:
        out.append(_apply_perturbation(cognitive_map, op, rng))
    return out


def _apply_perturbation(cmap: Dict[str, Any], op: str,
                        rng: random.Random) -> Dict[str, Any]:
    import copy
    perturbed = copy.deepcopy(cmap)

    bounds = perturbed.get("bounds", [-1.0, -1.0, 1.0, 1.0])
    xmin, ymin, xmax, ymax = bounds
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    cams = perturbed.get("cameras", [])
    if not cams:
        return perturbed

    if op == "mirror_x":
        for cam in cams:
            px, py = cam["position_xy"]
            cam["position_xy"] = [float(2 * cx - px), float(py)]
            cam["yaw_deg"] = float(180.0 - cam.get("yaw_deg", 0.0))
    elif op == "mirror_y":
        for cam in cams:
            px, py = cam["position_xy"]
            cam["position_xy"] = [float(px), float(2 * cy - py)]
            cam["yaw_deg"] = float(-cam.get("yaw_deg", 0.0))
    elif op == "rotate_180":
        for cam in cams:
            px, py = cam["position_xy"]
            cam["position_xy"] = [float(2 * cx - px), float(2 * cy - py)]
            cam["yaw_deg"] = float(cam.get("yaw_deg", 0.0) + 180.0)
    elif op == "shuffle":
        positions = [list(c["position_xy"]) for c in cams]
        rng.shuffle(positions)
        for cam, new_pos in zip(cams, positions):
            cam["position_xy"] = [float(new_pos[0]), float(new_pos[1])]
    else:  # no-op fallback
        pass

    return perturbed