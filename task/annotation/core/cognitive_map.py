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

        out_map: Dict[str, Any] = {
            "grid_size": [self.grid_size, self.grid_size],
            "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "cells": cells,
            "cameras": cameras_out,
            "objects": objects_out,
        }

        # Optional per-task reasoning overlay (e.g. MMSI camera-camera).
        # The builder does NOT interpret its content — it just forwards the
        # dict so the renderer can draw task-specific annotations on top of
        # the BEV. Keep it None when absent to avoid polluting serialized maps.
        overlay = (context.extra or {}).get("reasoning_overlay") if context else None
        if overlay:
            out_map["reasoning_overlay"] = overlay

        return out_map

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
            # OpenCV camera-frame basis rotated to world (columns of R):
            #   R[:,0] = camera +X (right) in world
            #   R[:,1] = camera +Y (down)  in world
            #   R[:,2] = camera +Z (forward) in world
            # Project both onto the world xy plane for BEV use. Note that
            # with arbitrary roll/pitch these projections are NOT exactly
            # orthogonal; using them verbatim (instead of reconstructing
            # "right" from yaw) is what keeps B's BEV position consistent
            # with the A-local (dx, dz) that defines the answer.
            forward_world = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
            right_world = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
            yaw_rad = math.atan2(forward_world[1], forward_world[0])
            out.append({
                "view_idx": int(vi),
                "position_xy": np.array([pos_xy[0], pos_xy[1]], dtype=float),
                "yaw_deg": math.degrees(yaw_rad),
                "forward_xy": np.array(
                    [float(forward_world[0]), float(forward_world[1])],
                    dtype=float),
                "right_xy": np.array(
                    [float(right_world[0]), float(right_world[1])],
                    dtype=float),
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

    # ── MindCube format conversion ────────────────────────────────────

    @staticmethod
    def _yaw_to_facing(yaw_deg: float, bounds: List[float]) -> Optional[str]:
        """Convert a world-frame yaw angle to a MindCube grid direction word.

        Grid coordinate system (MindCube convention):
          - World +X  →  grid col increases  →  "right"
          - World +Y  →  grid row increases  →  "down"
          - [0,0] is top-left, [G-1, G-1] is bottom-right.

        Quantized into 8 directions (cardinal + ordinal). Each sector spans
        45°, centered on the nominal direction:

          yaw ≈   0°  → right
          yaw ≈  45°  → down-right
          yaw ≈  90°  → down
          yaw ≈ 135°  → down-left
          yaw ≈ 180°  → left
          yaw ≈ 225°  → up-left
          yaw ≈ 270°  → up
          yaw ≈ 315°  → up-right

        The 4 cardinal words (up/down/left/right) are a strict subset of the
        previous 4-way output, so callers that only check these 4 remain
        correct; callers that want the finer granularity get the 4 diagonals
        in addition.
        """
        # 8 sectors, each 45° wide, boundaries at ±22.5° around each center.
        # Shift by +22.5° so that sector boundaries align with 22.5°, 67.5°, …
        angle = (yaw_deg + 22.5) % 360.0
        sector = int(angle // 45.0)  # 0..7
        return (
            "right",       # sector 0: centered at 0°
            "down-right",  # sector 1: centered at 45°
            "down",        # sector 2: centered at 90°
            "down-left",   # sector 3: centered at 135°
            "left",        # sector 4: centered at 180°
            "up-left",     # sector 5: centered at 225°
            "up",          # sector 6: centered at 270°
            "up-right",    # sector 7: centered at 315°
        )[sector]

    @staticmethod
    def to_mindcube_format(internal_map: Dict[str, Any],
                           view_index_to_image_num: Optional[Dict[int, int]] = None
                           ) -> Optional[Dict[str, Any]]:
        """Convert internal cognitive map dict to MindCube JSON format.

        MindCube format::

            {
              "objects": [
                {"name": "chair", "position": [col, row], "facing": "left"},
                {"name": "table", "position": [col, row]}
              ],
              "views": [
                {"name": "Image 1", "position": [col, row], "facing": "up"}
              ]
            }

        - position is [col, row] (i.e. [x, y] in grid coords), range [0, G-1].
        - [0,0] = top-left, [G-1, G-1] = bottom-right.
        - facing uses cardinal words: up/down/left/right.

        Args:
            internal_map: dict returned by ``build()``.
            view_index_to_image_num: optional mapping from SceneGraph view_idx
                to 1-based image number for display (e.g. {42: 1, 87: 2}).
                If None, cameras are numbered sequentially starting from 1.

        Returns:
            MindCube-format dict, or None if input is None.
        """
        if internal_map is None:
            return None

        bounds = internal_map.get("bounds", [0, 0, 1, 1])

        objects_out = []
        for obj in internal_map.get("objects", []):
            row, col = obj["cell_idx"]
            entry: Dict[str, Any] = {
                "name": obj["tag"],
                "position": [col, row],  # MindCube uses [col, row]
            }
            facing = CognitiveMapBuilder._yaw_to_facing(obj.get("yaw_deg", 0.0), bounds)
            if facing is not None and obj.get("yaw_deg", 0.0) != 0.0:
                entry["facing"] = facing
            objects_out.append(entry)

        views_out = []
        for i, cam in enumerate(internal_map.get("cameras", [])):
            row, col = cam["cell_idx"]
            if view_index_to_image_num is not None:
                img_num = view_index_to_image_num.get(cam["view_idx"], i + 1)
            else:
                img_num = i + 1
            entry: Dict[str, Any] = {
                "name": f"Image {img_num}",
                "position": [col, row],
                "view_idx": int(cam["view_idx"]),
                "yaw_deg": float(cam.get("yaw_deg", 0.0)),
            }
            # Forward/right unit-ish vectors projected to world xy, used by
            # reasoning overlays to draw truly axis-consistent arrows.
            fwd_xy = cam.get("forward_xy")
            if fwd_xy is not None:
                entry["forward_xy"] = [float(fwd_xy[0]), float(fwd_xy[1])]
            right_xy = cam.get("right_xy")
            if right_xy is not None:
                entry["right_xy"] = [float(right_xy[0]), float(right_xy[1])]
            facing = CognitiveMapBuilder._yaw_to_facing(cam.get("yaw_deg", 0.0), bounds)
            if facing is not None:
                entry["facing"] = facing
            views_out.append(entry)

        result: Dict[str, Any] = {
            "objects": objects_out,
            "views": views_out,
        }
        # Forward reasoning overlay verbatim — the renderer understands it.
        overlay = internal_map.get("reasoning_overlay")
        if overlay:
            result["reasoning_overlay"] = overlay
            # Expose world bounds under a *private* key so the reasoning
            # overlay drawer can map world-xy (e.g. object positions) back
            # to grid-cells using the same anisotropic scaling as this
            # converter. We use ``_render_bounds`` instead of ``bounds``
            # because ``CognitiveMapRenderer._render_impl`` treats the
            # presence of ``bounds`` as the signature of the legacy
            # *internal* format and would route this map to the wrong
            # renderer.
            result["_render_bounds"] = [float(bounds[0]), float(bounds[1]),
                                        float(bounds[2]), float(bounds[3])]
        return result


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

    def __init__(self, figsize=(7.5, 7.5), dpi=120, arrow_length_ratio=0.08,
                 max_question_lines: int = 3, max_chars_per_line: int = 80):
        # figsize is constrained to be square so the rendered PNG is square
        # regardless of whether a Q/A text panel is attached. The BEV subplot
        # and the optional text panel share the same square canvas via a
        # height-ratio gridspec (see ``_render_impl``).
        side = float(max(figsize[0], figsize[1]))
        self.figsize = (side, side)
        self.dpi = dpi
        self.arrow_length_ratio = arrow_length_ratio
        self.max_question_lines = max_question_lines
        self.max_chars_per_line = max_chars_per_line

    # ── public ────────────────────────────────────────────────────────

    def render(self, cognitive_map: Dict[str, Any], question: str = "",
               answer: str = "",
               draw_reasoning_overlay: bool = True) -> Optional[bytes]:
        """Render the cognitive map. Returns PNG bytes or None on failure.

        Args:
            draw_reasoning_overlay: 当为 True 时，会在 BEV 上叠加
                task-specific reasoning overlay（例如 MMSI cam-cam 的 A/B
                徽章、A 的 +Z_A/+X_A 局部坐标轴、A→B 虚线、dx/dz 分量箭头），
                并在文本面板附加 World / A-local 推理文本；当为 False 时只保留
                QA + 相机 + 物体这类"基础"元素，方便输出一份"干净版"图。
        """
        if cognitive_map is None:
            return None
        try:
            return self._render_impl(cognitive_map, question, answer,
                                     draw_reasoning_overlay=draw_reasoning_overlay)
        except Exception:
            # Never let rendering errors break the pipeline.
            return None

    def render_bev_only(self, cognitive_map: Dict[str, Any],
                        title: str = "",
                        draw_reasoning_overlay: bool = True) -> Optional[bytes]:
        """Render the BEV part only (used for MCQ option diagrams)."""
        if cognitive_map is None:
            return None
        try:
            return self._render_impl(cognitive_map, question="", answer="",
                                     bev_only=True, title=title,
                                     draw_reasoning_overlay=draw_reasoning_overlay)
        except Exception:
            return None

    # ── internals ─────────────────────────────────────────────────────

    # Direction word → (dx, dy) in grid coords where +x = right, +y = down.
    # Diagonals are unit-length so arrow lengths look consistent with the
    # cardinal arrows in the BEV render.
    _DIAG = math.sqrt(0.5)
    _FACING_VECTORS = {
        "up":         ( 0,     -1),
        "down":       ( 0,      1),
        "left":       (-1,      0),
        "right":      ( 1,      0),
        "up-right":   ( _DIAG, -_DIAG),
        "up-left":    (-_DIAG, -_DIAG),
        "down-right": ( _DIAG,  _DIAG),
        "down-left":  (-_DIAG,  _DIAG),
    }

    def _render_impl(self, cmap: Dict[str, Any], question: str, answer: str,
                     *, bev_only: bool = False, title: str = "",
                     draw_reasoning_overlay: bool = True) -> bytes:
        """Render a MindCube-format cognitive map on a grid-cell canvas.

        Accepts *either* the MindCube format (has "views" key) or the legacy
        internal format (has "cameras" + "bounds" keys). The BEV-pose-estimation
        task still passes the internal format via ``render_bev_only``.

        ``draw_reasoning_overlay`` controls whether task-specific reasoning
        annotations (e.g. MMSI cam-cam A/B axes/dx/dz arrows + world/local
        reasoning text) are overlaid on top of the BEV. The legacy internal
        format currently has no reasoning overlay, so the flag is only
        consumed by the MindCube renderer.
        """
        _, plt, Rectangle = _get_matplotlib()

        is_mindcube = "views" in cmap and "bounds" not in cmap

        if is_mindcube:
            return self._render_mindcube(
                cmap, question, answer,
                bev_only=bev_only, title=title,
                plt=plt, Rectangle=Rectangle,
                draw_reasoning_overlay=draw_reasoning_overlay,
            )
        else:
            return self._render_internal(cmap, question, answer,
                                         bev_only=bev_only, title=title,
                                         plt=plt, Rectangle=Rectangle)

    def _render_mindcube(self, cmap, question, answer, *,
                         bev_only, title, plt, Rectangle,
                         draw_reasoning_overlay: bool = True) -> bytes:
        """Render a MindCube-format map on a 10×10 grid-cell canvas."""
        g = 10  # grid size

        # For MMSI camera-camera, re-project the 10×10 grid into an
        # A-centered, A-forward-up **top-down** frame so that:
        #   • A sits at the grid center (5, 5)
        #   • A's +Z_A (forward) always points to the screen's +up direction
        #   • A's +X_A (right)   always points to the screen's +right direction
        # This eliminates the "BEV looks upside-down / mirrored" confusion and
        # lets the 8-sector diagram and the answer word line up 1:1 with the
        # rendered position of B. The re-projection only touches PNG rendering
        # for this specific task; the cogmap JSON written upstream is unchanged.
        #
        # NOTE: we intentionally do NOT gate reprojection on
        # ``draw_reasoning_overlay``. The A-forward-up frame is the canonical
        # layout for MMSI cam-cam QA — it must also apply to the "basic" PNG
        # (no overlay) so the rendered position of B matches the answer
        # word; otherwise the basic/full pair would disagree on which
        # quadrant B sits in.
        overlay_for_reproj = cmap.get("reasoning_overlay") if isinstance(cmap, dict) else None
        reprojected = False
        if (isinstance(overlay_for_reproj, dict)
                and overlay_for_reproj.get("kind") == "mmsi_cam_cam"):
            reprojected_cmap = self._reproject_to_a_forward_up(cmap, g)
            if reprojected_cmap is not None:
                cmap = reprojected_cmap
                reprojected = True

        if bev_only:
            fig = plt.figure(figsize=(self.figsize[0], self.figsize[0]),
                             dpi=self.dpi)
            ax = fig.add_subplot(1, 1, 1)
            ax_text = None
        else:
            # Give the text panel more vertical room — the reasoning overlay
            # for MMSI cam-cam tasks now emits up to ~6 lines (World section
            # + A-local section), which overflows the old 4:1 split.
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.6], hspace=0.25)
            ax = fig.add_subplot(gs[0, 0])
            ax_text = fig.add_subplot(gs[1, 0])

        fig.patch.set_facecolor(self.BACKGROUND)

        # Grid-cell coordinate system: x = col (right), y = row (down).
        # We set axis limits so cell centers sit at (col+0.5, row+0.5).
        ax.set_xlim(0, g)
        ax.set_ylim(g, 0)  # invert Y so row 0 is at top
        ax.set_aspect("equal", adjustable="box")
        if reprojected:
            # The cmap has been reprojected so that +X_A maps to +col and
            # +Z_A maps to -row (screen up). That matches a BEV "looking
            # INTO the screen" convention, which is the opposite of how a
            # reader naturally inspects a 2D top-down picture — they look
            # OUT of the screen, so their own right corresponds to A's left
            # and vice-versa. To align the rendered picture with the
            # "reader looks out of the screen" convention (and with how the
            # MMSI answer word reads), mirror the x-axis. This flips every
            # data-coordinate artist drawn later (ray lines, sector labels,
            # +X_A arrow, B marker, dx/dz components) in one place, so the
            # classifier answer, the sector B sits in, and the +X_A arrow
            # all agree visually.
            ax.invert_xaxis()
            # After invert_xaxis: +X_A (A-right) now points to the SCREEN's
            # left, and +Z_A (A-forward) still points to screen up (because
            # y is independently inverted via set_ylim(g, 0)).
            ax.set_xlabel("← +X_A (A-right)   |   BEV: reader looks OUT of screen")
            ax.set_ylabel("+Z_A (↑ A-forward)")
        else:
            ax.set_xlabel("col (→ right)")
            ax.set_ylabel("row (↓ down)")
        if title:
            ax.set_title(title, fontsize=10)

        # Draw grid lines.
        for i in range(g + 1):
            ax.plot([i, i], [0, g], color=self.GRID_COLOR,
                    linewidth=0.6, alpha=0.6, zorder=1)
            ax.plot([0, g], [i, i], color=self.GRID_COLOR,
                    linewidth=0.6, alpha=0.6, zorder=1)

        # Draw objects.
        arrow_len = 0.35
        for obj in cmap.get("objects", []):
            col, row = obj["position"]
            cx, cy = col + 0.5, row + 0.5
            color = self.OBJECT_COLOR
            ax.plot(cx, cy, marker="s", color=color, markersize=8, zorder=4)
            ax.text(cx, cy - 0.35, str(obj.get("name", "")),
                    fontsize=7, ha="center", va="bottom", color=color,
                    fontweight="bold", zorder=5)
            facing = obj.get("facing")
            if facing and facing in self._FACING_VECTORS:
                fdx, fdy = self._FACING_VECTORS[facing]
                ax.annotate(
                    "", xy=(cx + fdx * arrow_len, cy + fdy * arrow_len),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.4),
                    zorder=5,
                )

        # Draw views (cameras).
        for view in cmap.get("views", []):
            col, row = view["position"]
            cx, cy = col + 0.5, row + 0.5
            ax.plot(cx, cy, marker="o", color=self.CAMERA_COLOR,
                    markersize=7, zorder=5)
            ax.text(cx + 0.15, cy - 0.15,
                    str(view.get("name", "")),
                    fontsize=7, color=self.CAMERA_COLOR, zorder=6)
            facing = view.get("facing")
            if facing and facing in self._FACING_VECTORS:
                fdx, fdy = self._FACING_VECTORS[facing]
                ax.annotate(
                    "", xy=(cx + fdx * arrow_len, cy + fdy * arrow_len),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=self.CAMERA_COLOR,
                                    linewidth=1.6),
                    zorder=6,
                )

        # ── Reasoning overlay (task-specific; e.g. MMSI cam-cam direction) ──
        # When the caller explicitly asks for a "clean" image (QA + camera +
        # object only), skip all overlay drawing AND leave the reasoning text
        # block empty so the bottom panel only shows Q / A.
        if draw_reasoning_overlay:
            reasoning_text = self._draw_reasoning_overlay(ax, cmap, g)
        else:
            reasoning_text = ""

        # Question / answer text panel.
        if ax_text is not None:
            ax_text.axis("off")
            wrapped_q = self._wrap_text(question or "")
            q_display = "\n".join(wrapped_q[: self.max_question_lines]) \
                if wrapped_q else ""
            # Layout (top→bottom): Q block, A block, then reasoning block.
            # y coords are in axes fraction; va="top" so each anchors at its y.
            ax_text.text(0.0, 1.00, f"Q: {q_display}",
                         fontsize=9, ha="left", va="top",
                         transform=ax_text.transAxes, wrap=True)
            ax_text.text(0.0, 0.72, f"A: {answer or ''}",
                         fontsize=10, ha="left", va="top",
                         transform=ax_text.transAxes,
                         color="#b8860b", fontweight="bold", wrap=True)
            if reasoning_text:
                ax_text.text(0.0, 0.55, reasoning_text,
                             fontsize=7, ha="left", va="top",
                             transform=ax_text.transAxes,
                             color="#444444", family="monospace",
                             linespacing=1.25)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches=None,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _render_internal(self, cmap, question, answer, *,
                         bev_only, title, plt, Rectangle) -> bytes:
        """Render the legacy internal-format map (used by BEV pose estimation)."""
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

        ax_bev.set_xlim(xmin, xmax)
        ax_bev.set_ylim(ymin, ymax)
        ax_bev.set_aspect("equal", adjustable="box")
        ax_bev.set_xlabel("world X (m)")
        ax_bev.set_ylabel("world Y (m)")
        if title:
            ax_bev.set_title(title, fontsize=10)

        for i in range(gx + 1):
            x = xmin + (xmax - xmin) * i / gx
            ax_bev.plot([x, x], [ymin, ymax], color=self.GRID_COLOR,
                        linewidth=0.6, alpha=0.6, zorder=1)
        for j in range(gy + 1):
            y = ymin + (ymax - ymin) * j / gy
            ax_bev.plot([xmin, xmax], [y, y], color=self.GRID_COLOR,
                        linewidth=0.6, alpha=0.6, zorder=1)

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
        fig.savefig(buf, format="png", bbox_inches=None,
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

    # ── Reasoning overlay ─────────────────────────────────────────────

    # Colors used by the reasoning overlay. Chosen to be distinct from the
    # base camera/object/anchor colors.
    _REASON_AXIS_FWD_COLOR = "#8b5cf6"   # purple : A's forward (+Z_A)
    _REASON_AXIS_RIGHT_COLOR = "#ea580c" # orange : A's right  (+X_A)
    _REASON_LINE_COLOR = "#000000"       # black  : A → B connector
    _REASON_COMP_COLOR = "#6b7280"       # gray   : dx / dz components

    # ── mmsi_obj_face_obj_obj colors ──────────────────────────────────
    # Anchor object = purple fill (matches the +Z "forward" family).
    # Orienting object (the one we face) = green star (matches the
    # camera-facing-object-camera overlay's object marker).
    # Query object = red star (the MCQ's actual target).
    _REASON_ANCHOR_COLOR = "#7c3aed"      # violet-600
    _REASON_ORIENTING_COLOR = "#16a34a"   # green-600
    _REASON_ORIENTING_FILL = "#bbf7d0"    # green-200 (star fill)
    _REASON_QUERY_COLOR = "#b91c1c"       # red-700
    _REASON_QUERY_FILL = "#fecaca"        # red-200 (star fill)

    def _draw_reasoning_overlay_obj_face_obj_obj(
        self, ax, cmap: Dict[str, Any], g: int, overlay: Dict[str, Any]
    ) -> str:
        """Draw MMSI obj-face-obj-obj reasoning annotations on the BEV axes.

        Expected schema in ``overlay``::

            {
              "kind": "mmsi_obj_face_obj_obj",
              "view_idx": int,
              "anchor_world_xy":    [wx, wy],   # reference OBJECT (origin)
              "orienting_world_xy": [wx, wy],   # the object we face
              "query_world_xy":     [wx, wy],   # the MCQ target
              "anchor_tag": str, "orienting_tag": str, "query_tag": str,
              "virtual_fwd_xy":   [ux, uy],     # unit, anchor → orienting
              "virtual_right_xy": [ux, uy],     # unit, fwd rotated CW 90°
              "dx": float,   # query.xy in anchor-local (+right, metres)
              "dz": float,   # query.xy in anchor-local (+forward, metres)
              "answer": str,
            }

        Layout drawn:
          * 8-sector quadrant diagram anchored at the anchor object
            (rays in the anchor-local frame, matching the question).
          * +Z (purple) axis toward the orienting object; +X (orange)
            axis to the right of that facing direction.
          * Dashed green connector anchor → orienting (the "facing"
            heading) with a green star on the orienting object.
          * Dashed black connector anchor → query (the queried vector)
            with a red star on the query object.
          * dx (orange) / dz (purple) components as thin arrows along
            the local axes.

        Returns the reasoning text for the side panel.
        """
        # ── Resolve BEV bounds & world→grid scaling ────────────────────
        bounds = cmap.get("bounds")
        if bounds is None:
            bounds = cmap.get("_render_bounds")
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4
                and (bounds[2] - bounds[0]) > 1e-6
                and (bounds[3] - bounds[1]) > 1e-6):
            # Without bounds we cannot place world-xy on the grid.
            return ""
        xmin_w = float(bounds[0]); ymin_w = float(bounds[1])
        xmax_w = float(bounds[2]); ymax_w = float(bounds[3])
        col_per_wx = float(g) / (xmax_w - xmin_w)
        row_per_wy = float(g) / (ymax_w - ymin_w)

        def _w_to_grid(wx: float, wy: float) -> Tuple[float, float]:
            return (wx - xmin_w) * col_per_wx, (wy - ymin_w) * row_per_wy

        def _world_dir_to_grid(wx: float, wy: float) -> Tuple[float, float]:
            gx = wx * col_per_wx
            gy = wy * row_per_wy
            n = math.hypot(gx, gy) or 1.0
            return gx / n, gy / n

        # ── Pull overlay fields ─────────────────────────────────────────
        try:
            a_wxy = overlay["anchor_world_xy"]
            o_wxy = overlay["orienting_world_xy"]
            q_wxy = overlay["query_world_xy"]
            vf = overlay["virtual_fwd_xy"]
            vr = overlay["virtual_right_xy"]
            a_wx, a_wy = float(a_wxy[0]), float(a_wxy[1])
            o_wx, o_wy = float(o_wxy[0]), float(o_wxy[1])
            q_wx, q_wy = float(q_wxy[0]), float(q_wxy[1])
            fwd_wx, fwd_wy = float(vf[0]), float(vf[1])
            right_wx, right_wy = float(vr[0]), float(vr[1])
        except (KeyError, TypeError, ValueError, IndexError):
            return ""

        # Renormalise fwd / right in world-xy (should already be unit).
        fn = math.hypot(fwd_wx, fwd_wy) or 1.0
        fwd_wx /= fn; fwd_wy /= fn
        rn = math.hypot(right_wx, right_wy) or 1.0
        right_wx /= rn; right_wy /= rn

        anchor_tag = str(overlay.get("anchor_tag") or "anchor")
        orient_tag = str(overlay.get("orienting_tag") or "orient")
        query_tag = str(overlay.get("query_tag") or "query")

        # ── Anchor BEV position ─────────────────────────────────────────
        ax_cx, ax_cy = _w_to_grid(a_wx, a_wy)

        # ── 8-sector quadrant diagram anchored at the anchor object ────
        sector_names = [
            f"Front (→ {orient_tag})",  # center   0°
            "Front-Right",              # center  +45°
            "Right",                    # center  +90°
            "Back-Right",               # center +135°
            "Back",                     # center ±180°
            "Back-Left",                # center -135°
            "Left",                     # center  -90°
            "Front-Left",               # center  -45°
        ]
        sector_centers_deg = [0.0, 45.0, 90.0, 135.0, 180.0,
                              -135.0, -90.0, -45.0]
        boundary_angles_deg = [-157.5, -112.5, -67.5, -22.5,
                               22.5, 67.5, 112.5, 157.5]

        # Grid-space forward / right (anisotropically scaled once).
        fwd_gx, fwd_gy = _world_dir_to_grid(fwd_wx, fwd_wy)
        right_gx, right_gy = _world_dir_to_grid(right_wx, right_wy)

        ray_len = 16.0
        quadrant_line_color = "#94a3b8"   # slate-400
        quadrant_label_color = "#64748b"  # slate-500
        for ang_deg in boundary_angles_deg:
            ang_rad = math.radians(ang_deg)
            wx = math.sin(ang_rad) * right_wx + math.cos(ang_rad) * fwd_wx
            wy = math.sin(ang_rad) * right_wy + math.cos(ang_rad) * fwd_wy
            vx, vy = _world_dir_to_grid(wx, wy)
            ax.plot(
                [ax_cx, ax_cx + vx * ray_len],
                [ax_cy, ax_cy + vy * ray_len],
                color=quadrant_line_color,
                linewidth=0.8,
                linestyle=(0, (4, 3)),
                alpha=0.55,
                zorder=2,
            )

        # Sector-name labels along each sector's center ray.
        label_radius = 3.2
        for name, ang_deg in zip(sector_names, sector_centers_deg):
            ang_rad = math.radians(ang_deg)
            wx = math.sin(ang_rad) * right_wx + math.cos(ang_rad) * fwd_wx
            wy = math.sin(ang_rad) * right_wy + math.cos(ang_rad) * fwd_wy
            vx, vy = _world_dir_to_grid(wx, wy)
            lx = ax_cx + vx * label_radius
            ly = ax_cy + vy * label_radius
            lx = max(0.3, min(float(g) - 0.3, lx))
            ly = max(0.3, min(float(g) - 0.3, ly))
            ax.text(
                lx, ly, name,
                fontsize=6.5,
                color=quadrant_label_color,
                ha="center", va="center",
                alpha=0.85,
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.12",
                          fc="white", ec="none", alpha=0.65),
            )

        # ── Anchor / orienting / query object markers ──────────────────
        # Anchor: violet filled circle + letter "A".
        ax.plot(
            [ax_cx], [ax_cy],
            marker="o", markersize=11,
            markerfacecolor="#ede9fe",           # violet-100
            markeredgecolor=self._REASON_ANCHOR_COLOR,
            markeredgewidth=1.4,
            linestyle="None", zorder=8,
        )
        ax.text(
            ax_cx, ax_cy, "A",
            fontsize=9, color=self._REASON_ANCHOR_COLOR,
            fontweight="bold", ha="center", va="center",
            zorder=9,
        )
        ax.text(
            ax_cx + 0.35, ax_cy - 0.35, anchor_tag,
            fontsize=7, color=self._REASON_ANCHOR_COLOR,
            ha="left", va="bottom", zorder=9,
            bbox=dict(boxstyle="round,pad=0.12",
                      fc="white", ec="none", alpha=0.85),
        )

        # Orienting: dashed green heading line + green star.
        o_cx, o_cy = _w_to_grid(o_wx, o_wy)
        if 0.0 <= o_cx <= float(g) and 0.0 <= o_cy <= float(g):
            ax.plot(
                [ax_cx, o_cx], [ax_cy, o_cy],
                color=self._REASON_ORIENTING_COLOR, linewidth=1.3,
                linestyle=(0, (3, 2)), alpha=0.85, zorder=6,
            )
            ax.plot(
                [o_cx], [o_cy],
                marker="*", markersize=14,
                markerfacecolor=self._REASON_ORIENTING_FILL,
                markeredgecolor=self._REASON_ORIENTING_COLOR,
                markeredgewidth=1.2,
                linestyle="None", zorder=8,
            )
            ax.text(
                o_cx + 0.15, o_cy - 0.35, f"O: {orient_tag}",
                fontsize=7, color=self._REASON_ORIENTING_COLOR,
                ha="left", va="bottom", zorder=9,
                bbox=dict(boxstyle="round,pad=0.12",
                          fc="white", ec="none", alpha=0.85),
            )

        # Query: dashed black connector + red star.
        q_cx, q_cy = _w_to_grid(q_wx, q_wy)
        if 0.0 <= q_cx <= float(g) and 0.0 <= q_cy <= float(g):
            ax.annotate(
                "", xy=(q_cx, q_cy), xytext=(ax_cx, ax_cy),
                arrowprops=dict(arrowstyle="->",
                                color=self._REASON_LINE_COLOR,
                                linewidth=1.2, linestyle="dashed"),
                zorder=7,
            )
            ax.plot(
                [q_cx], [q_cy],
                marker="*", markersize=14,
                markerfacecolor=self._REASON_QUERY_FILL,
                markeredgecolor=self._REASON_QUERY_COLOR,
                markeredgewidth=1.2,
                linestyle="None", zorder=8,
            )
            ax.text(
                q_cx + 0.15, q_cy - 0.35, f"Q: {query_tag}",
                fontsize=7, color=self._REASON_QUERY_COLOR,
                ha="left", va="bottom", zorder=9,
                bbox=dict(boxstyle="round,pad=0.12",
                          fc="white", ec="none", alpha=0.85),
            )

        # ── Anchor-local axes arrows ───────────────────────────────────
        axis_len = 0.9
        # +Z (forward = toward orienting) — purple.
        ax.annotate(
            "", xy=(ax_cx + fwd_gx * axis_len, ax_cy + fwd_gy * axis_len),
            xytext=(ax_cx, ax_cy),
            arrowprops=dict(arrowstyle="->",
                            color=self._REASON_AXIS_FWD_COLOR,
                            linewidth=1.8),
            zorder=7,
        )
        ax.text(ax_cx + fwd_gx * (axis_len + 0.15),
                ax_cy + fwd_gy * (axis_len + 0.15),
                f"+Z (face→{orient_tag})", fontsize=7,
                color=self._REASON_AXIS_FWD_COLOR, zorder=8)

        # +X (right of facing direction) — orange.
        ax.annotate(
            "", xy=(ax_cx + right_gx * axis_len,
                    ax_cy + right_gy * axis_len),
            xytext=(ax_cx, ax_cy),
            arrowprops=dict(arrowstyle="->",
                            color=self._REASON_AXIS_RIGHT_COLOR,
                            linewidth=1.8),
            zorder=7,
        )
        ax.text(ax_cx + right_gx * (axis_len + 0.15),
                ax_cy + right_gy * (axis_len + 0.15),
                "+X (right of face)", fontsize=7,
                color=self._REASON_AXIS_RIGHT_COLOR, zorder=8)

        # ── dx / dz component arrows (thin gray) ───────────────────────
        try:
            dx = float(overlay.get("dx", 0.0))
            dz = float(overlay.get("dz", 0.0))
        except (TypeError, ValueError):
            dx, dz = 0.0, 0.0

        max_mag = max(abs(dx), abs(dz), 1e-3)
        scale = min(1.5 / max_mag, 0.5)
        if abs(dz) > 1e-3:
            ax.annotate(
                "",
                xy=(ax_cx + fwd_gx * dz * scale,
                    ax_cy + fwd_gy * dz * scale),
                xytext=(ax_cx, ax_cy),
                arrowprops=dict(arrowstyle="->",
                                color=self._REASON_COMP_COLOR,
                                linewidth=1.2, alpha=0.75),
                zorder=6,
            )
            ax.text(ax_cx + fwd_gx * dz * scale * 0.55,
                    ax_cy + fwd_gy * dz * scale * 0.55,
                    f"dz={dz:+.2f}m", fontsize=7,
                    color=self._REASON_COMP_COLOR, zorder=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="none", alpha=0.75))
        if abs(dx) > 1e-3:
            ax.annotate(
                "",
                xy=(ax_cx + right_gx * dx * scale,
                    ax_cy + right_gy * dx * scale),
                xytext=(ax_cx, ax_cy),
                arrowprops=dict(arrowstyle="->",
                                color=self._REASON_COMP_COLOR,
                                linewidth=1.2, alpha=0.75),
                zorder=6,
            )
            ax.text(ax_cx + right_gx * dx * scale * 0.55,
                    ax_cy + right_gy * dx * scale * 0.55,
                    f"dx={dx:+.2f}m", fontsize=7,
                    color=self._REASON_COMP_COLOR, zorder=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="none", alpha=0.75))

        # ── Reasoning text for the side panel ──────────────────────────
        answer_word = str(overlay.get("answer", ""))
        angle_deg = math.degrees(math.atan2(dx, dz)) if (dx or dz) else 0.0
        world_lines = [
            "World (xy = horizontal plane, z = up):",
            f"  A[anchor={anchor_tag}]=({a_wx:+.2f}, {a_wy:+.2f})m  "
            f"O[orient={orient_tag}]=({o_wx:+.2f}, {o_wy:+.2f})m",
            f"  Q[query={query_tag}]=({q_wx:+.2f}, {q_wy:+.2f})m",
        ]
        local_lines = [
            f"Virtual heading: stand at {anchor_tag}, face {orient_tag}.",
            "Local frame on horizontal plane "
            "(forward=+Z=anchor→orient, right=+X=forward CW 90°):",
            f"  dx={dx:+.2f}m (right of face)  "
            f"dz={dz:+.2f}m (front of face)",
            f"  angle=atan2(dx,dz)={angle_deg:+6.1f}°  "
            f"sector→ {answer_word}",
        ]
        return "\n".join(world_lines + local_lines)

    def _draw_reasoning_overlay(self, ax, cmap: Dict[str, Any], g: int
                                ) -> str:
        """Draw MMSI cam-cam reasoning annotations on the BEV axes.

        Expected schema in ``cmap["reasoning_overlay"]``::

            {
              "kind": "mmsi_cam_cam",
              "anchor_view_idx": int,   # camera A (reference frame)
              "target_view_idx": int,   # camera B (queried)
              "dx": float,              # B.x in A-local (meters, +right)
              "dz": float,              # B.z in A-local (meters, +forward)
              "answer": str,            # classified direction word
            }

        The function locates A and B in ``cmap["views"]`` by ``view_idx``,
        then draws:
          * A's local axes (forward = purple, right = orange)
          * A→B connector (dashed black)
          * dx component along A's right axis, dz along A's forward axis
          * A/B letter labels so the viewer knows which is which

        Returns a short reasoning string to be displayed in the text panel,
        or an empty string when no overlay is drawn.
        """
        overlay = cmap.get("reasoning_overlay") if isinstance(cmap, dict) else None
        if not overlay:
            return ""
        kind = overlay.get("kind")
        if kind not in ("mmsi_cam_cam", "mmsi_cam_face_obj_cam",
                        "mmsi_obj_face_obj_obj"):
            return ""

        # ── Branch off for the purely object-anchored variant ────────
        # `mmsi_obj_face_obj_obj` has *no* camera-A / camera-B structure:
        # the reference point is an anchor OBJECT (world-xy), forward is
        # the anchor→orienting direction, and the queried entity is a
        # third (query) object. That makes the rest of this function
        # (which assumes `anchor_view_idx` / `target_view_idx` live in
        # `cmap["views"]`) inapplicable. Delegate to a dedicated helper
        # that speaks the object-anchored schema natively.
        if kind == "mmsi_obj_face_obj_obj":
            return self._draw_reasoning_overlay_obj_face_obj_obj(
                ax, cmap, g, overlay
            )

        # When the anchor frame is a *virtual* one (A standing at Camera A
        # but facing the target object rather than A's real optical axis),
        # the overlay must draw the 8-sector diagram around that virtual
        # heading. The producer passes the unit world-xy vectors for the
        # virtual forward/right under ``virtual_fwd_xy`` / ``virtual_right_xy``
        # — we apply them below, after A's grid position has been looked up.
        is_virtual = (kind == "mmsi_cam_face_obj_cam")
        # Detect A-forward-up reprojected view (set by _reproject_to_a_forward_up).
        # In this view: A is at grid center (5, 5); forward_xy = (0, -1) in
        # grid-space means "+Z_A → screen up"; right_xy = (1, 0) means
        # "+X_A → screen right". The overlay below can then be drawn with
        # fixed directions, bypassing the anisotropic world→grid scaling.
        is_local_view = bool(cmap.get("_a_forward_up_view"))

        # Build a view_idx → (col+0.5, row+0.5, yaw_deg_grid) table.
        view_lut: Dict[int, Tuple[float, float, float]] = {}
        for v in cmap.get("views", []):
            vi = v.get("view_idx")
            if vi is None:
                continue
            col, row = v["position"]
            # yaw_deg stored on the view is in *world* frame (atan2(fy, fx)
            # with world +Y = down in grid). The grid axes are col=+X_world,
            # row=+Y_world, so world yaw maps straight to grid yaw — no flip
            # needed for the arrow direction because `_render_mindcube`
            # inverted the y-axis via ylim only for display.
            view_lut[int(vi)] = (col + 0.5, row + 0.5,
                                 float(v.get("yaw_deg", 0.0)),
                                 v.get("forward_xy"),
                                 v.get("right_xy"))

        try:
            a_vi = int(overlay["anchor_view_idx"])
            b_vi = int(overlay["target_view_idx"])
        except (KeyError, TypeError, ValueError):
            return ""
        if a_vi not in view_lut or b_vi not in view_lut:
            return ""

        ax_cx, ax_cy, a_yaw, a_fwd_xy, a_right_xy = view_lut[a_vi]
        bx_cx, bx_cy, _, _, _ = view_lut[b_vi]

        # Per-axis world → grid scale factors.
        # The BEV uses an *anisotropic* mapping — col spans the world-x range
        # [xmin, xmax] and row spans the world-y range [ymin, ymax] with
        # INDEPENDENT scale factors (see ``_xy_to_cell``). If those ranges
        # differ (typical for corridor-like scenes) a 1-metre vector along
        # world-+x renders as a *different* number of grid cells than a
        # 1-metre vector along world-+y. The 8-sector diagram and the
        # +Z_A / +X_A arrows must apply the same anisotropic scaling,
        # otherwise the on-screen angle will not match the A-local (dx, dz)
        # used by ``_classify_direction``.
        #
        # When ``is_local_view`` is True, the cmap has already been
        # re-projected into the A-centered, A-forward-up top-down frame,
        # so the world-space frame is orthonormal and the scaling is 1:1.
        bounds = cmap.get("bounds") if isinstance(cmap, dict) else None
        if bounds is None and isinstance(cmap, dict):
            # ``_reproject_to_a_forward_up`` stashes the effective render
            # bounds under a private key so the format auto-detection in
            # ``_render_impl`` (which keys off "bounds") is not confused.
            bounds = cmap.get("_render_bounds")
        if (bounds is not None and len(bounds) == 4
                and (bounds[2] - bounds[0]) > 1e-6
                and (bounds[3] - bounds[1]) > 1e-6):
            xmin_w, ymin_w, xmax_w, ymax_w = (
                float(bounds[0]), float(bounds[1]),
                float(bounds[2]), float(bounds[3]))
            col_per_wx = float(g) / (xmax_w - xmin_w)
            row_per_wy = float(g) / (ymax_w - ymin_w)
        else:
            col_per_wx = 1.0
            row_per_wy = 1.0

        def _world_dir_to_grid(wx: float, wy: float) -> Tuple[float, float]:
            """Apply the same anisotropic scaling as ``_xy_to_cell`` to a
            direction vector, then renormalise so downstream code can scale
            it by a fixed grid-cell length."""
            gx = wx * col_per_wx
            gy = wy * row_per_wy
            n = math.hypot(gx, gy) or 1.0
            return gx / n, gy / n

        # A's world-frame forward/right vectors (unit, on the xy plane).
        # Keep these raw so the 8-sector diagram can synthesise A-local
        # directions *in world space* first and then apply the anisotropic
        # world→grid scaling exactly once.
        #
        # For the virtual "facing the object" variant we do NOT use A's
        # real optical axis — the overlay provides the virtual forward
        # (A → object, projected on world-xy) and the virtual right
        # (forward rotated clockwise 90°) directly. This is the whole
        # point of the task variant: the 8-sector diagram must be drawn
        # around the *virtual* heading, not A's camera axis.
        if is_virtual:
            _vf = overlay.get("virtual_fwd_xy")
            _vr = overlay.get("virtual_right_xy")
            if isinstance(_vf, (list, tuple)) and len(_vf) >= 2:
                a_fwd_xy = (float(_vf[0]), float(_vf[1]))
            else:
                a_fwd_xy = None
            if isinstance(_vr, (list, tuple)) and len(_vr) >= 2:
                a_right_xy = (float(_vr[0]), float(_vr[1]))
            else:
                a_right_xy = None

        if a_fwd_xy is not None:
            fwd_wx = float(a_fwd_xy[0])
            fwd_wy = float(a_fwd_xy[1])
            _n = math.hypot(fwd_wx, fwd_wy) or 1.0
            fwd_wx /= _n
            fwd_wy /= _n
        else:
            a_yaw_rad = math.radians(a_yaw)
            fwd_wx = math.cos(a_yaw_rad)
            fwd_wy = math.sin(a_yaw_rad)

        if a_right_xy is not None:
            right_wx = float(a_right_xy[0])
            right_wy = float(a_right_xy[1])
            _n = math.hypot(right_wx, right_wy) or 1.0
            right_wx /= _n
            right_wy /= _n
        else:
            # Legacy fallback: derive right from yaw assuming level camera.
            a_yaw_rad = math.radians(a_yaw)
            right_wx = math.sin(a_yaw_rad)
            right_wy = -math.cos(a_yaw_rad)

        # Grid-space versions of forward/right (anisotropically scaled and
        # renormalised) — consumed by the +Z_A / +X_A axis arrows drawn
        # further below.
        fwd_gx, fwd_gy = _world_dir_to_grid(fwd_wx, fwd_wy)
        right_gx, right_gy = _world_dir_to_grid(right_wx, right_wy)

        # ── 8-sector quadrant diagram anchored at A ──────────────────
        # Draws 8 dashed rays emanating from A, partitioning the BEV into
        # the same 8 sectors used by ``_classify_direction``:
        #   Front / Front-Right / Right / Back-Right /
        #   Back  / Back-Left  / Left  / Front-Left
        # Sector boundaries are at ±22.5°, ±67.5°, ±112.5°, ±157.5° relative
        # to A's forward (+Z_A) axis, measured by
        #   angle = atan2(dx, dz) with dx along +X_A (right), dz along +Z_A.
        # A direction vector at angle θ in A-local coords maps to grid space
        # via:  v_grid = sin(θ) * right + cos(θ) * forward
        # so that θ=0 points along +Z_A (Front), θ=+90° along +X_A (Right).
        sector_names = [
            "Front",        # center  0°, bounds [-22.5°, +22.5°)
            "Front-Right",  # center +45°
            "Right",        # center +90°
            "Back-Right",   # center +135°
            "Back",         # center ±180°
            "Back-Left",    # center -135°
            "Left",         # center -90°
            "Front-Left",   # center -45°
        ]
        if is_virtual:
            # Annotate "Front" so the reader immediately sees which way
            # "facing the object" points. The other 7 sector names are
            # always relative to "Front" so they need no change.
            facing_tag = str(overlay.get("facing_object_tag") or "obj")
            sector_names = [
                f"Front (→ {facing_tag})",  # center 0°
                "Front-Right",
                "Right",
                "Back-Right",
                "Back",
                "Back-Left",
                "Left",
                "Front-Left",
            ]
        # Sector center angles in A-local coords (degrees), aligned with the
        # names above.
        sector_centers_deg = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]
        # Sector boundary angles (every 45° offset by 22.5°).
        boundary_angles_deg = [-157.5, -112.5, -67.5, -22.5,
                               22.5, 67.5, 112.5, 157.5]

        # Ray length: long enough to span the visible 10×10 grid from A.
        # Using 16 cells guarantees the rays reach all four edges regardless
        # of where A sits inside the grid.
        ray_len = 16.0

        quadrant_line_color = "#94a3b8"  # slate-400: subtle gray-blue
        quadrant_label_color = "#64748b"  # slate-500

        for ang_deg in boundary_angles_deg:
            ang_rad = math.radians(ang_deg)
            # Compose the direction *in world space* first so that the
            # anisotropic world→grid scaling (col_per_wx, row_per_wy) is
            # applied exactly once, matching ``_xy_to_cell``:
            #   world_dir = sin(θ)·right_world + cos(θ)·forward_world
            #   grid_dir  = (world_dir.x · col_per_wx,
            #               world_dir.y · row_per_wy)   then renormalised
            wx = math.sin(ang_rad) * right_wx + math.cos(ang_rad) * fwd_wx
            wy = math.sin(ang_rad) * right_wy + math.cos(ang_rad) * fwd_wy
            vx, vy = _world_dir_to_grid(wx, wy)
            ax.plot(
                [ax_cx, ax_cx + vx * ray_len],
                [ax_cy, ax_cy + vy * ray_len],
                color=quadrant_line_color,
                linewidth=0.8,
                linestyle=(0, (4, 3)),  # dashed
                alpha=0.55,
                zorder=2,
            )

        # Sector-name labels placed along each sector's center ray. Distance
        # is chosen so labels stay on-canvas for most A positions; clamp to
        # grid bounds to avoid drawing outside the visible 10×10 area.
        label_radius = 3.2  # grid cells from A along the sector center
        for name, ang_deg in zip(sector_names, sector_centers_deg):
            ang_rad = math.radians(ang_deg)
            wx = math.sin(ang_rad) * right_wx + math.cos(ang_rad) * fwd_wx
            wy = math.sin(ang_rad) * right_wy + math.cos(ang_rad) * fwd_wy
            vx, vy = _world_dir_to_grid(wx, wy)
            lx = ax_cx + vx * label_radius
            ly = ax_cy + vy * label_radius
            # Clamp label position so it stays inside [0.3, g-0.3].
            lx = max(0.3, min(float(g) - 0.3, lx))
            ly = max(0.3, min(float(g) - 0.3, ly))
            ax.text(
                lx, ly, name,
                fontsize=6.5,
                color=quadrant_label_color,
                ha="center", va="center",
                alpha=0.85,
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.12",
                          fc="white", ec="none", alpha=0.65),
            )

        # Highlight A and B with letter labels (in addition to the "Image N"
        # labels already drawn).
        ax.text(ax_cx - 0.15, ax_cy + 0.35, "A",
                fontsize=11, color="#111827", fontweight="bold",
                ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.15",
                          fc="#fde68a", ec="#111827", lw=0.8))
        ax.text(bx_cx + 0.15, bx_cy + 0.35, "B",
                fontsize=11, color="#111827", fontweight="bold",
                ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.15",
                          fc="#bfdbfe", ec="#111827", lw=0.8))

        # Target-object marker for the virtual "facing the object" variant.
        # We plot a green star at the object's BEV cell plus a slim green
        # line from A toward it, so the reader can verify that A's +Z
        # (facing) arrow indeed points to the object.
        if is_virtual:
            obj_wxy = overlay.get("object_world_xy")
            if (isinstance(obj_wxy, (list, tuple)) and len(obj_wxy) >= 2
                    and bounds is not None and len(bounds) == 4):
                try:
                    obj_wx, obj_wy = float(obj_wxy[0]), float(obj_wxy[1])
                    xmin_w = float(bounds[0]); ymin_w = float(bounds[1])
                    obj_cx = (obj_wx - xmin_w) * col_per_wx
                    obj_cy = (obj_wy - ymin_w) * row_per_wy
                    # Only draw if the object falls inside the visible grid.
                    if (0.0 <= obj_cx <= float(g) and
                            0.0 <= obj_cy <= float(g)):
                        facing_tag = str(
                            overlay.get("facing_object_tag") or "obj"
                        )
                        # Dashed green line A → object (the virtual heading).
                        ax.plot(
                            [ax_cx, obj_cx], [ax_cy, obj_cy],
                            color="#16a34a", linewidth=1.3,
                            linestyle=(0, (3, 2)), alpha=0.85,
                            zorder=6,
                        )
                        # Green star at the object.
                        ax.plot(
                            [obj_cx], [obj_cy],
                            marker="*", markersize=14,
                            markerfacecolor="#bbf7d0",
                            markeredgecolor="#166534",
                            markeredgewidth=1.2,
                            linestyle="None", zorder=8,
                        )
                        ax.text(
                            obj_cx + 0.15, obj_cy - 0.35, facing_tag,
                            fontsize=7, color="#166534",
                            ha="left", va="bottom", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.12",
                                      fc="white", ec="none", alpha=0.85),
                        )
                except (TypeError, ValueError):
                    pass

        # A's local axes in grid coordinates.
        # Grid axes: +col = +X_world, +row = +Y_world. So any world-xy
        # vector (wx, wy) maps directly to (col_delta, row_delta) = (wx, wy).
        # fwd_gx / fwd_gy / right_gx / right_gy were computed earlier (before
        # the 8-sector quadrant diagram), so they are reused here.

        axis_len = 0.9   # grid cells

        # A's forward axis (purple, solid).
        ax.annotate(
            "", xy=(ax_cx + fwd_gx * axis_len, ax_cy + fwd_gy * axis_len),
            xytext=(ax_cx, ax_cy),
            arrowprops=dict(arrowstyle="->",
                            color=self._REASON_AXIS_FWD_COLOR,
                            linewidth=1.8),
            zorder=7,
        )
        # Forward axis label — for the virtual "facing the object" variant
        # it is the A→object direction; otherwise it is A's real camera
        # optical axis.
        if is_virtual:
            facing_tag = str(overlay.get("facing_object_tag") or "obj")
            _fwd_label = f"+Z (face→{facing_tag})"
        else:
            _fwd_label = "+Z_A (forward)"
        ax.text(ax_cx + fwd_gx * (axis_len + 0.15),
                ax_cy + fwd_gy * (axis_len + 0.15),
                _fwd_label, fontsize=7,
                color=self._REASON_AXIS_FWD_COLOR, zorder=8)

        # A's right axis (orange, solid).
        ax.annotate(
            "", xy=(ax_cx + right_gx * axis_len, ax_cy + right_gy * axis_len),
            xytext=(ax_cx, ax_cy),
            arrowprops=dict(arrowstyle="->",
                            color=self._REASON_AXIS_RIGHT_COLOR,
                            linewidth=1.8),
            zorder=7,
        )
        _right_label = "+X (right of face)" if is_virtual else "+X_A (right)"
        ax.text(ax_cx + right_gx * (axis_len + 0.15),
                ax_cy + right_gy * (axis_len + 0.15),
                _right_label, fontsize=7,
                color=self._REASON_AXIS_RIGHT_COLOR, zorder=8)

        # Connector A → B (dashed black).
        ax.annotate(
            "", xy=(bx_cx, bx_cy), xytext=(ax_cx, ax_cy),
            arrowprops=dict(arrowstyle="->", color=self._REASON_LINE_COLOR,
                            linewidth=1.2, linestyle="dashed"),
            zorder=7,
        )

        # Decompose B - A into dx·right + dz·forward components, drawn as
        # thin gray arrows starting at A. Use the actual (dx, dz) from the
        # overlay so the arrow lengths match the numeric reasoning even
        # under pose/yaw drift.
        try:
            dx = float(overlay.get("dx", 0.0))
            dz = float(overlay.get("dz", 0.0))
        except (TypeError, ValueError):
            dx, dz = 0.0, 0.0

        # Scale meters → grid cells. Pick so the longer of |dx|/|dz| ≈ 1.5 cells,
        # capped so the arrows don't cover the whole map.
        max_mag = max(abs(dx), abs(dz), 1e-3)
        scale = min(1.5 / max_mag, 0.5)

        # dz along A's forward axis.
        if abs(dz) > 1e-3:
            ax.annotate(
                "",
                xy=(ax_cx + fwd_gx * dz * scale, ax_cy + fwd_gy * dz * scale),
                xytext=(ax_cx, ax_cy),
                arrowprops=dict(arrowstyle="->",
                                color=self._REASON_COMP_COLOR,
                                linewidth=1.2, alpha=0.75),
                zorder=6,
            )
            ax.text(ax_cx + fwd_gx * dz * scale * 0.55,
                    ax_cy + fwd_gy * dz * scale * 0.55,
                    f"dz={dz:+.2f}m", fontsize=7,
                    color=self._REASON_COMP_COLOR, zorder=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="none", alpha=0.75))

        # dx along A's right axis.
        if abs(dx) > 1e-3:
            ax.annotate(
                "",
                xy=(ax_cx + right_gx * dx * scale,
                    ax_cy + right_gy * dx * scale),
                xytext=(ax_cx, ax_cy),
                arrowprops=dict(arrowstyle="->",
                                color=self._REASON_COMP_COLOR,
                                linewidth=1.2, alpha=0.75),
                zorder=6,
            )
            ax.text(ax_cx + right_gx * dx * scale * 0.55,
                    ax_cy + right_gy * dx * scale * 0.55,
                    f"dx={dx:+.2f}m", fontsize=7,
                    color=self._REASON_COMP_COLOR, zorder=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="none", alpha=0.75))

        # Reasoning text for the side panel.
        answer_word = str(overlay.get("answer", ""))
        angle_deg = math.degrees(math.atan2(dx, dz)) if (dx or dz) else 0.0

        # World-frame (horizontal plane) info — optional, only emitted when
        # the producer attached it. Gives viewers the absolute positions of
        # both cameras + their Δxy, which sets the stage for the A-local
        # decomposition below.
        world_lines: List[str] = []
        a_wxy = overlay.get("a_world_xy")
        b_wxy = overlay.get("b_world_xy")
        d_wxy = overlay.get("delta_world_xy")
        a_yaw_world = overlay.get("a_yaw_world_deg")
        if (isinstance(a_wxy, (list, tuple)) and len(a_wxy) >= 2 and
                isinstance(b_wxy, (list, tuple)) and len(b_wxy) >= 2):
            try:
                ax_m, ay_m = float(a_wxy[0]), float(a_wxy[1])
                bx_m, by_m = float(b_wxy[0]), float(b_wxy[1])
                if isinstance(d_wxy, (list, tuple)) and len(d_wxy) >= 2:
                    dxw, dyw = float(d_wxy[0]), float(d_wxy[1])
                else:
                    dxw, dyw = bx_m - ax_m, by_m - ay_m
                dist_xy = math.hypot(dxw, dyw)
                yaw_str = (f"  A_yaw_world={float(a_yaw_world):+6.1f}°"
                           if a_yaw_world is not None else "")
                world_lines.append(
                    "World (xy = horizontal plane, z = up):"
                )
                world_lines.append(
                    f"  A=({ax_m:+.2f}, {ay_m:+.2f})m  "
                    f"B=({bx_m:+.2f}, {by_m:+.2f})m"
                )
                world_lines.append(
                    f"  ΔXY=({dxw:+.2f}, {dyw:+.2f})m  "
                    f"|ΔXY|={dist_xy:.2f}m{yaw_str}"
                )
            except (TypeError, ValueError):
                world_lines = []

        local_lines = [
            "A's local frame on horizontal plane "
            "(forward=+Z_xy, right=+X_xy; world-z ignored):",
            f"  dx={dx:+.2f}m (right)  dz={dz:+.2f}m (front)",
            f"  angle=atan2(dx,dz)={angle_deg:+6.1f}°  "
            f"sector→ {answer_word}",
        ]
        if is_virtual:
            facing_tag = str(overlay.get("facing_object_tag") or "obj")
            local_lines = [
                f"Virtual heading: A stands at Camera A, faces {facing_tag}.",
                "Local frame on horizontal plane "
                "(forward=+Z=A→obj, right=+X=forward rotated CW 90°):",
                f"  dx={dx:+.2f}m (right of face)  "
                f"dz={dz:+.2f}m (front of face)",
                f"  angle=atan2(dx,dz)={angle_deg:+6.1f}°  "
                f"sector→ {answer_word}",
            ]

        if world_lines:
            return "\n".join(world_lines + local_lines)
        return "\n".join(local_lines)

    # ── A-forward-up re-projection (MMSI cam-cam only) ────────────────

    def _reproject_to_a_forward_up(self, cmap: Dict[str, Any], g: int
                                   ) -> Optional[Dict[str, Any]]:
        """Re-project a MindCube cmap into an A-centered, A-forward-up,
        **top-down** view so the rendered BEV matches human intuition:

        * A sits at the grid center ``(g/2, g/2)``.
        * A's +Z_A (forward) points to the screen's +up direction.
        * A's +X_A (right)   points to the screen's +right direction.

        The re-projection is driven entirely by A's world-space
        ``forward_xy`` / ``right_xy`` (unit-ish vectors on the xy plane) and
        the *relative* grid offset of every entity to A. The resulting cmap
        uses the same 10×10 grid so downstream drawing code keeps working,
        but with coordinates living in A's local frame instead of world
        frame.

        Returns ``None`` if A cannot be located or lacks basis vectors.
        """
        overlay = cmap.get("reasoning_overlay") if isinstance(cmap, dict) else None
        if not overlay or overlay.get("kind") != "mmsi_cam_cam":
            return None

        try:
            a_vi = int(overlay["anchor_view_idx"])
        except (KeyError, TypeError, ValueError):
            return None

        # Locate A and read its world-xy basis.
        a_view = None
        for v in cmap.get("views", []):
            if v.get("view_idx") is not None and int(v["view_idx"]) == a_vi:
                a_view = v
                break
        if a_view is None:
            return None

        fwd_xy = a_view.get("forward_xy")
        right_xy = a_view.get("right_xy")
        if fwd_xy is None or right_xy is None:
            return None
        try:
            fwd_wx = float(fwd_xy[0]); fwd_wy = float(fwd_xy[1])
            rgt_wx = float(right_xy[0]); rgt_wy = float(right_xy[1])
        except (TypeError, ValueError, IndexError):
            return None

        # Normalise. These are the unit vectors of A's local frame projected
        # onto the world xy plane.
        fn = math.hypot(fwd_wx, fwd_wy)
        rn = math.hypot(rgt_wx, rgt_wy)
        if fn < 1e-9 or rn < 1e-9:
            return None
        fwd_wx /= fn; fwd_wy /= fn
        rgt_wx /= rn; rgt_wy /= rn

        # Anisotropic grid scaling currently embedded in the MindCube
        # positions: col = (wx - xmin) / (xmax - xmin) * g, and similarly
        # for row. To invert a relative grid offset back into a relative
        # world offset we need (dx_w, dy_w) = (dcol / col_per_wx,
        # drow / row_per_wy). Those scale factors are stored on the cmap
        # only in the *internal* format; MindCube strips them. We derive
        # them from the cmap's (optional) `_render_bounds` if present,
        # otherwise fall back to an isotropic 1 cell = 1 m assumption
        # (which is fine because the reprojection below only consumes
        # ratios of lengths along forward/right — the absolute scale
        # cancels out when we rescale to fit the target 10×10 grid).
        bounds = cmap.get("_render_bounds") or cmap.get("bounds")
        if (isinstance(bounds, (list, tuple)) and len(bounds) == 4 and
                (bounds[2] - bounds[0]) > 1e-6 and (bounds[3] - bounds[1]) > 1e-6):
            col_per_wx = float(g) / (float(bounds[2]) - float(bounds[0]))
            row_per_wy = float(g) / (float(bounds[3]) - float(bounds[1]))
        else:
            col_per_wx = 1.0
            row_per_wy = 1.0

        # A's grid position (used as the re-projection origin).
        a_col, a_row = a_view["position"]
        a_cx = float(a_col) + 0.5
        a_cy = float(a_row) + 0.5

        def _grid_to_local_uv(gx: float, gy: float) -> Tuple[float, float]:
            """Map a grid-space point (cell-center coords) into A's local
            (u, v) frame where:
              * u = signed distance along A-right (world units)
              * v = signed distance along A-forward (world units)
            """
            # Grid offset from A (in cells), inverted to world-xy offset.
            dwx = (gx - a_cx) / col_per_wx
            dwy = (gy - a_cy) / row_per_wy
            # Project onto A's right and forward basis vectors.
            u = dwx * rgt_wx + dwy * rgt_wy
            v = dwx * fwd_wx + dwy * fwd_wy
            return u, v

        # Pass 1: collect (u, v) for every drawable entity so we can pick a
        # uniform scale factor that makes all of them fit within the 10×10
        # grid with A at its center.
        uv_points: List[Tuple[float, float]] = []
        for v in cmap.get("views", []):
            col, row = v["position"]
            uv_points.append(_grid_to_local_uv(col + 0.5, row + 0.5))
        for o in cmap.get("objects", []):
            col, row = o["position"]
            uv_points.append(_grid_to_local_uv(col + 0.5, row + 0.5))

        # Also seed with dx/dz from the overlay so B is visible even when
        # only a subset of views are carried in `views`.
        try:
            dx_m = float(overlay.get("dx", 0.0))
            dz_m = float(overlay.get("dz", 0.0))
            uv_points.append((dx_m, dz_m))
        except (TypeError, ValueError):
            pass

        max_abs = 0.0
        for u, vv in uv_points:
            max_abs = max(max_abs, abs(u), abs(vv))
        # Leave a 1-cell margin on each side: scale so max_abs maps to
        # g/2 - 1 cells. Cap scale to avoid blowing up degenerate cases.
        if max_abs < 1e-6:
            scale = 1.0
        else:
            scale = (float(g) / 2.0 - 1.0) / max_abs

        def _local_uv_to_new_grid(u: float, vv: float) -> Tuple[float, float]:
            """Place (u, v) onto the new 10×10 grid centered on A.
            - +u (A-right)   → +col (screen right)
            - +v (A-forward) → -row (screen up, since ylim is inverted)
            """
            new_cx = float(g) / 2.0 + u * scale
            new_cy = float(g) / 2.0 - vv * scale
            return new_cx, new_cy

        def _cxcy_to_position(new_cx: float, new_cy: float) -> List[int]:
            """Convert a cell-center (cx, cy) back to integer [col, row]."""
            col = int(math.floor(new_cx))
            row = int(math.floor(new_cy))
            col = max(0, min(g - 1, col))
            row = max(0, min(g - 1, row))
            return [col, row]

        # Build new views / objects lists with re-projected positions and
        # A-local forward/right basis vectors.
        import copy
        new_views: List[Dict[str, Any]] = []
        for v in cmap.get("views", []):
            nv = copy.deepcopy(v)
            col, row = v["position"]
            u, vv = _grid_to_local_uv(col + 0.5, row + 0.5)
            ncx, ncy = _local_uv_to_new_grid(u, vv)
            nv["position"] = _cxcy_to_position(ncx, ncy)

            # Re-express each view's forward / right in A's local frame.
            f_xy = v.get("forward_xy")
            r_xy = v.get("right_xy")
            if f_xy is not None:
                fwx = float(f_xy[0]); fwy = float(f_xy[1])
                # Project onto (right, forward) basis → (+u_component, +v_component)
                nu = fwx * rgt_wx + fwy * rgt_wy
                nv_ = fwx * fwd_wx + fwy * fwd_wy
                # In the new grid frame: +u → +grid-x, +v → -grid-y (y is inverted)
                nv["forward_xy"] = [nu, -nv_]
            if r_xy is not None:
                rwx = float(r_xy[0]); rwy = float(r_xy[1])
                nu = rwx * rgt_wx + rwy * rgt_wy
                nv_ = rwx * fwd_wx + rwy * fwd_wy
                nv["right_xy"] = [nu, -nv_]

            # yaw_deg in the new frame: atan2(new_fy_grid, new_fx_grid)
            # Grid convention: yaw measured in (col, row) plane.
            if "forward_xy" in nv:
                nfx, nfy = nv["forward_xy"]
                nv["yaw_deg"] = math.degrees(math.atan2(nfy, nfx))
                # The pre-computed `facing` word (e.g. "up"/"down-right") was
                # derived from the old yaw in world frame; with the axes
                # re-oriented it would now point the wrong way. Recompute it
                # from the new grid-space yaw using the same 8-sector
                # quantiser used by the builder.
                try:
                    nv["facing"] = CognitiveMapBuilder._yaw_to_facing(
                        nv["yaw_deg"], [0.0, 0.0, float(g), float(g)]
                    )
                except Exception:
                    nv.pop("facing", None)
            new_views.append(nv)

        new_objects: List[Dict[str, Any]] = []
        for o in cmap.get("objects", []):
            no = copy.deepcopy(o)
            col, row = o["position"]
            u, vv = _grid_to_local_uv(col + 0.5, row + 0.5)
            ncx, ncy = _local_uv_to_new_grid(u, vv)
            no["position"] = _cxcy_to_position(ncx, ncy)
            # The pre-computed `facing` word was derived from world-frame yaw;
            # after re-orienting axes to A-local it would point the wrong way.
            # MMSI cam-cam QA never queries object orientation, so simply
            # drop the stale facing rather than trying to re-rotate it.
            no.pop("facing", None)
            new_objects.append(no)

        # Assemble the re-projected cmap. Keep the reasoning overlay
        # verbatim so dx/dz (already expressed in A-local frame) render
        # correctly, and stash private keys for downstream helpers.
        new_cmap = dict(cmap)  # shallow copy of top-level keys
        new_cmap["views"] = new_views
        new_cmap["objects"] = new_objects
        new_cmap["_a_forward_up_view"] = True
        # In the re-projected frame each grid cell == `1/scale` world metres
        # along both axes → isotropic. Expose this via _render_bounds so
        # ``_draw_reasoning_overlay`` computes col_per_wx = row_per_wy, and
        # the 8-sector diagram / +Z_A / +X_A arrows come out aligned with
        # the screen axes regardless of the original anisotropy.
        new_cmap["_render_bounds"] = [0.0, 0.0, float(g), float(g)]
        return new_cmap


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