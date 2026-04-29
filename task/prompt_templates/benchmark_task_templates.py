"""
Prompt templates for the benchmark-style multiview tasks (relative distance,
clockwise, camera movement, MMSI quartet, BEV pose, manipulation viewpoint).

These templates are *not* currently wired into the corresponding
``AnnotationGenerator`` classes (which build their prompts inline for clarity
and determinism), but are registered here so that:

1. The TemplateRegistry exposes a consistent naming scheme for downstream
   audits / dashboards, matching the existing template-file convention.
2. Future refactors can switch any task over to ``render_prompt`` without
   introducing new template files.

All template names follow the pattern ``<task>.<variant>``.
"""

# ─── Relative distance (all-angles style) ─────────────────────────────
relative_distance_questions = [
    "Considering all views, is [A] closer to [B] or to [C]? [Y]",
    "Taking every viewpoint into account, is [A] closer to [B] or [C]? [Y]",
]
relative_distance_answers = ["[X]"]

# ─── Camera clockwise (BLINK style) ───────────────────────────────────
clockwise_questions = [
    "Is the camera moving clockwise around the [A]? [Y]",
    "From the viewer's perspective, does the camera rotate clockwise around "
    "the [A]? [Y]",
]
clockwise_answers = ["[X]"]

# ─── Camera movement direction (VSI-Bench style) ──────────────────────
camera_movement_questions = [
    "From the start to the end of the video, the camera mainly moved in "
    "which direction? [Y]",
]
camera_movement_answers = ["[X]"]

# ─── MMSI Camera–Camera ───────────────────────────────────────────────
mmsi_camera_camera_questions = [
    "In image 2, the camera is located in which direction relative to image "
    "1's camera? [Y]",
]
mmsi_camera_camera_answers = ["[X]"]

# ─── MMSI Camera–Object ───────────────────────────────────────────────
# Two wording variants, matching the two sub-tasks emitted by
# ``mmsi_camera_object.py``:
#   * cross: object visible only in image 2, asked w.r.t. image 1
#   * self:  object visible in image 1, asked w.r.t. image 1 itself
# Both resolve to the same 8-option MCQ answer space.
mmsi_camera_object_questions = [
    "In image 2, where is the [A] located relative to the camera that "
    "took image 1? [Y]",
    "In image 1, where is the [A] located relative to the camera that "
    "took image 1? [Y]",
]
mmsi_camera_object_answers = ["[X]"]

# ─── MMSI Object–Object ───────────────────────────────────────────────
mmsi_object_object_questions = [
    "Considering both images, is the [A] to the left or right of the [B] in "
    "the world frame defined by the line connecting the two cameras? [Y]",
]
mmsi_object_object_answers = ["[X]"]

# ─── MMSI Camera Motion (composite) ───────────────────────────────────
mmsi_camera_motion_questions = [
    "Between image 1 and image 2, the camera mainly moved... [Y]",
]
mmsi_camera_motion_answers = ["[X]"]

# ─── BEV pose estimation (all-angles style) ───────────────────────────
bev_pose_questions = [
    "Below are 3 RGB views (views 1, 2, 3) followed by 4 bird's-eye-view "
    "diagrams labeled A-D. Which BEV diagram correctly represents the "
    "spatial layout of the three cameras? [Y]",
]
bev_pose_answers = ["[X]"]

# ─── Manipulation viewpoint (all-angles style approximation) ──────────
manipulation_view_questions = [
    "Comparing View 1 and View 2 (which show a camera viewpoint difference "
    "around the [A], not a physical object manipulation), what apparent "
    "change is observed? [Y]",
]
manipulation_view_answers = ["[X]"]


# ─── Template Registration ────────────────────────────────────────────
from ..annotation.core.prompt_template import TemplateRegistry, PromptTemplate

TemplateRegistry.register("relative_distance.mcq", PromptTemplate(
    questions=relative_distance_questions,
    answers=relative_distance_answers,
))
TemplateRegistry.register("clockwise.yes_no", PromptTemplate(
    questions=clockwise_questions,
    answers=clockwise_answers,
))
TemplateRegistry.register("camera_movement.mcq", PromptTemplate(
    questions=camera_movement_questions,
    answers=camera_movement_answers,
))
TemplateRegistry.register("mmsi.camera_camera", PromptTemplate(
    questions=mmsi_camera_camera_questions,
    answers=mmsi_camera_camera_answers,
))
TemplateRegistry.register("mmsi.camera_object", PromptTemplate(
    questions=mmsi_camera_object_questions,
    answers=mmsi_camera_object_answers,
))
TemplateRegistry.register("mmsi.object_object", PromptTemplate(
    questions=mmsi_object_object_questions,
    answers=mmsi_object_object_answers,
))
TemplateRegistry.register("mmsi.camera_motion", PromptTemplate(
    questions=mmsi_camera_motion_questions,
    answers=mmsi_camera_motion_answers,
))
TemplateRegistry.register("bev_pose.mcq", PromptTemplate(
    questions=bev_pose_questions,
    answers=bev_pose_answers,
))
TemplateRegistry.register("manipulation_view.mcq", PromptTemplate(
    questions=manipulation_view_questions,
    answers=manipulation_view_answers,
))
