"""
Base class for singleview annotation tasks.

Extracts the shared run/apply_transform/create_messages patterns
from all 8 singleview annotation files.
"""

import os
import threading

from .scene_graph import SceneGraph
from .visual_marker import VisualMarker, MarkConfig
from .message_builder import create_singleview_messages
from .prompt_template import PromptTemplate, TemplateRegistry
from .cognitive_map import (
    CognitiveMapBuilder,
    CognitiveMapContext,
    CognitiveMapRenderer,
)
from .cognitive_map_config import parse_cognitive_map_settings

from task.base_task import BaseTask
from utils.point_cloud_utils import clean_point_cloud
from utils.image_utils import convert_pil_to_bytes

# Trigger template registration on first import of any annotation task
import task.prompt_templates  # noqa: F401


class BaseAnnotationTask(BaseTask):
    """
    Base class for all singleview annotation tasks.

    Subclasses must implement:
        - process(self, example) -> (prompts, processed_images, question_tags, question_types)

    Optionally override:
        - get_mark_config() -> MarkConfig
        - check_example(example) -> bool
        - create_messages_from_prompts(prompts, processed_images) -> list
    """

    QUESTION_TAG = "Unknown"
    SUB_TASKS = {}  # Subclass override: {"name": {"default": N, "handler": "_method_name"}}

    def __init__(self, args):
        super().__init__(args)
        self._thread_local = threading.local()
        self.scaling_factor = args.get("scaling_factor", 1)
        self.filter_tags = args.get("filter_tags", None)
        self._sub_tasks_config = self._parse_sub_tasks(args.get("sub_tasks", None))

        # Cognitive map feature (disabled by default for backward compat).
        self._cog_settings = parse_cognitive_map_settings(args)
        self._cog_builder = None
        self._cog_renderer = None
        self._cog_dump_counter = 0
        self._cog_fail_count = 0
        self._cog_total_count = 0
        if self._cog_settings.active:
            self._cog_builder = CognitiveMapBuilder(
                grid_size=self._cog_settings.grid_size,
                padding_ratio=self._cog_settings.padding_ratio,
            )
            if self._cog_settings.enable_visualization:
                self._cog_renderer = CognitiveMapRenderer()
        self._cog_output_dir = args.get("output_dir", None)
        self._cog_dump_lock = threading.Lock()

    def _parse_sub_tasks(self, raw):
        """Parse sub_tasks config from YAML.

        Supports:
            None             → use all defaults
            "all"            → use all defaults
            ["a", "b"]       → enable only these sub_tasks, use default counts
            {"a": 3, "b": 5} → enable these sub_tasks with specified counts

        Returns:
            None (use defaults) or dict {sub_task_name: count_or_None}
        """
        if raw is None or raw == "all":
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            return {k: None for k in raw}
        # Handle SimpleNamespace from YAML dict_to_namespace conversion
        if hasattr(raw, '__dict__'):
            return vars(raw)
        raise ValueError(f"Invalid sub_tasks config: {raw}")

    def get_sub_task_count(self, sub_task, default=1):
        """Get how many prompts to generate for a given sub_task.

        Returns:
            int: number of prompts. 0 means skip this sub_task.
        """
        if self._sub_tasks_config is None:
            return default
        if sub_task not in self._sub_tasks_config:
            return 0
        count = self._sub_tasks_config[sub_task]
        return default if count is None else int(count)

    @property
    def marker(self):
        """Thread-local VisualMarker. Each thread gets its own instance."""
        tl = self._thread_local
        if not hasattr(tl, 'marker'):
            tl.marker = VisualMarker(self.get_mark_config())
        return tl.marker

    @marker.setter
    def marker(self, value):
        self._thread_local.marker = value

    def get_mark_config(self) -> MarkConfig:
        """Override to provide task-specific mark configuration."""
        return MarkConfig()

    @staticmethod
    def _get_cloud(marked):
        """Extract (desc, pointcloud) from a marked result (desc, node)."""
        desc, node = marked
        cloud = node.view_appearances[0].pointcloud_camera
        return desc, cloud

    @staticmethod
    def _clean_cloud(cloud):
        """Remove statistical outliers from a pointcloud."""
        return clean_point_cloud(cloud)

    @staticmethod
    def _shuffle_mcq(candidates, correct_idx=0):
        """Shuffle candidates into A/B/C/D, return (shuffled, answer_letter)."""
        import random
        order = list(range(len(candidates)))
        random.shuffle(order)
        answer = "ABCD"[order.index(correct_idx)]
        return [candidates[j] for j in order], answer

    def mark_and_prompt(self, nodes, image, prompt_func, *,
                        each=False, mark_prob=1.0, prompt_args=None):
        """Mark nodes on image and generate prompts.

        Args:
            each: If True, call prompt_func(marked) per node → return list.
                  If False, call prompt_func(*all_marked, **prompt_args) → return single.
            mark_prob: Probability of applying visual marks (default 1.0 = always).
                       When skipped, uses (node.tag, node) as unmarked fallback.
            prompt_args: Extra kwargs forwarded to prompt_func.

        Returns:
            (prompt_or_prompts, processed_image)
        """
        import random

        if prompt_args is None:
            prompt_args = {}

        if random.random() < mark_prob:
            processed_image, marked = self.marker.mark_objects(image, nodes)
        else:
            processed_image = {"bytes": convert_pil_to_bytes(image)}
            marked = [(n.tag, n) for n in nodes]

        if each:
            prompts = [prompt_func(m, **prompt_args) for m in marked]
            return prompts, processed_image
        else:
            prompt = prompt_func(*marked, **prompt_args)
            return prompt, processed_image

    def check_example(self, example) -> bool:
        """Pre-check common required fields. Subclasses should call super() first."""
        if "image" not in example:
            return False
        if "obj_tags" not in example or len(example["obj_tags"]) == 0:
            return False
        return True

    def build_scene_graph(self, example) -> SceneGraph:
        """Build a SceneGraph from the example dict. Override for custom logic."""
        return SceneGraph.from_singleview_example(example)

    def process(self, graph, example):
        """Generic sub_task dispatch loop.

        Iterates over SUB_TASKS, calls each handler with count from config.
        Handler signature (preferred): _generate_xxx(self, graph)
            -> (prompt, image, qtype) | (prompt, image, qtype, cog_ctx)
            | list of such tuples | None

        The 4-tuple form attaches a CognitiveMapContext for the cognitive
        map feature. Handlers that return 3-tuples remain fully backward
        compatible.
        """
        prompts, images, qtypes, cog_contexts = [], [], [], []

        def _append(item):
            if not isinstance(item, tuple):
                raise ValueError(
                    f"Handler must return a tuple, got {type(item).__name__}"
                )
            if len(item) == 3:
                p, img, qt = item
                cog = None
            elif len(item) == 4:
                p, img, qt, cog = item
            else:
                raise ValueError(
                    f"Handler must return a 3- or 4-tuple, got length {len(item)}"
                )
            prompts.append(p)
            images.append(img)
            qtypes.append(qt)
            cog_contexts.append(cog)

        for name, meta in self.SUB_TASKS.items():
            count = self.get_sub_task_count(name, default=meta["default"])
            if count == 0:
                continue
            handler = getattr(self, meta["handler"])
            for _ in range(count):
                result = handler(graph)
                if result is None:
                    continue
                if isinstance(result, list):
                    for sub in result:
                        _append(sub)
                else:
                    _append(result)
        tags = [[self.QUESTION_TAG]] * len(prompts)
        return prompts, images, tags, qtypes, cog_contexts

    def get_template(self, name: str) -> PromptTemplate:
        """Get a template from the registry. Subclass can override for customization."""
        return TemplateRegistry.get(name)

    def render_prompt(self, template_name: str, condition: bool = None, *,
                      shared: dict = None, q_args: dict = None, a_args: dict = None) -> str:
        """One-step: get template → sample → fill → return 'question Answer: answer'."""
        tpl = self.get_template(template_name)
        return tpl.render(condition=condition, shared=shared, q_args=q_args, a_args=a_args)

    def create_messages_from_prompts(self, prompts, processed_images=None):
        """
        Default singleview message creation.
        Splits on "Answer: ", prepends "<image>" tag.
        """
        return create_singleview_messages(prompts)

    def apply_transform(self, example, idx=None):
        """Standard transform pipeline: check → build graph → process → create_messages → set fields.

        Thread-safe: each thread gets its own VisualMarker via the thread-local
        `self.marker` property, avoiding shared mutable color_queue state.
        """
        # Best-effort scene identifier for diagnostic prints.
        scene_id = example.get("scene_id") if isinstance(example, dict) else None
        task_name = self.__class__.__name__

        if not self.check_example(example):
            print(f"[apply_transform] {task_name} DROPPED by check_example  "
                  f"scene={scene_id}", flush=True)
            return None, False

        # Reset thread-local marker for this example
        self.marker = VisualMarker(self.get_mark_config())

        graph = self.build_scene_graph(example)
        process_result = self.process(graph, example)

        # Backward compat: process() may return 4- or 5-tuple.
        if len(process_result) == 5:
            prompts, processed_images, question_tags, question_types, cog_contexts = process_result
        elif len(process_result) == 4:
            prompts, processed_images, question_tags, question_types = process_result
            cog_contexts = [None] * len(prompts)
        else:
            raise ValueError(
                f"process() must return 4- or 5-tuple, got length {len(process_result)}"
            )

        print(f"[apply_transform] {task_name} process -> "
              f"prompts={len(prompts)}  images={len(processed_images)}  "
              f"qtypes={question_types}  scene={scene_id}", flush=True)

        if len(prompts) == 0:
            print(f"[apply_transform] {task_name} DROPPED prompts=0  "
                  f"scene={scene_id}", flush=True)
            return None, False

        messages = self.create_messages_from_prompts(prompts, processed_images)

        example["messages"] = messages
        example["QA_images"] = processed_images
        example["question_tags"] = question_tags
        example["question_types"] = question_types

        # Cognitive map attachment (only when enabled via YAML).
        if self._cog_settings.active:
            self._attach_cognitive_maps(example, graph, prompts, cog_contexts,
                                        question_tags)

        print(f"[apply_transform] {task_name} OK  messages={len(messages)}  "
              f"QA_images={len(processed_images)}  scene={scene_id}", flush=True)
        return example, True

    # ─── Cognitive Map Hooks ─────────────────────────────────────────────

    def _make_singleview_cog_context(self, graph, nodes=None, anchor_node_id=None):
        """Build a CognitiveMapContext from a singleview graph + object nodes.

        Helper used by singleview handlers to attach the standard context.
        """
        view_indices = list(graph.views.keys()) if hasattr(graph, "views") else [0]
        if nodes is None:
            node_ids = [n.node_id for n in graph.get_object_nodes()]
        else:
            node_ids = [n.node_id for n in nodes if n is not None]
        if not view_indices and not node_ids:
            return None
        return CognitiveMapContext(
            view_indices=view_indices,
            node_ids=node_ids,
            anchor_node_id=anchor_node_id,
        )

    def _attach_cognitive_maps(self, example, graph, prompts, cog_contexts,
                               question_tags):
        """Build cognitive maps for each generated QA and attach to example.

        The stored ``cognitive_maps`` list contains MindCube-format dicts
        (``{objects, views}`` with grid-cell positions and direction words)
        so they can be directly serialized into JSONL for downstream use.
        """
        from .cognitive_map import CognitiveMapBuilder

        maps = []
        images = []
        for i, prompt in enumerate(prompts):
            ctx = cog_contexts[i] if i < len(cog_contexts) else None
            cmap_internal = None
            cmap_mindcube = None
            cmap_img = None
            if ctx is not None and self._cog_builder is not None:
                try:
                    cmap_internal = self._cog_builder.build(graph, ctx)
                except Exception:
                    cmap_internal = None
            if cmap_internal is not None:
                # Build view_idx -> 1-based image number mapping from context.
                view_map = None
                if ctx is not None and ctx.view_indices:
                    view_map = {vi: idx + 1
                                for idx, vi in enumerate(ctx.view_indices)}
                cmap_mindcube = CognitiveMapBuilder.to_mindcube_format(
                    cmap_internal, view_index_to_image_num=view_map)
            # Render using MindCube format (grid-cell canvas).
            render_target = cmap_mindcube if cmap_mindcube is not None else cmap_internal
            if render_target is not None and self._cog_renderer is not None:
                q_text, a_text = self._split_question_answer(prompt)
                try:
                    cmap_img = self._cog_renderer.render(render_target, q_text, a_text)
                except Exception:
                    cmap_img = None
                with self._cog_dump_lock:
                    self._cog_total_count += 1
                    if cmap_img is None:
                        self._cog_fail_count += 1
                if cmap_img is not None and self._cog_settings.dump_samples:
                    tag = question_tags[i][0] if (i < len(question_tags)
                                                  and question_tags[i]) else "tag"
                    self._maybe_dump_sample(cmap_img, tag)
            maps.append(cmap_mindcube)
            images.append(cmap_img)
        example["cognitive_maps"] = maps
        example["cognitive_map_images"] = images

    @staticmethod
    def _split_question_answer(prompt):
        """Split a 'question Answer: answer' prompt into its halves."""
        if not isinstance(prompt, str):
            return "", ""
        marker = "Answer:"
        if marker in prompt:
            q, _, a = prompt.partition(marker)
            return q.strip(), a.strip()
        return prompt.strip(), ""

    def _maybe_dump_sample(self, png_bytes, tag):
        """Dump the first N PNGs to disk for quick inspection (best-effort)."""
        with self._cog_dump_lock:
            if self._cog_dump_counter >= self._cog_settings.dump_sample_count:
                return
            idx = self._cog_dump_counter
            self._cog_dump_counter += 1
        try:
            if not self._cog_output_dir:
                return
            out_dir = os.path.join(self._cog_output_dir,
                                   "cognitive_map_samples")
            os.makedirs(out_dir, exist_ok=True)
            safe_tag = "".join(c if c.isalnum() or c in "_-" else "_"
                               for c in str(tag))
            out_path = os.path.join(out_dir, f"{idx:04d}_{safe_tag}.png")
            with open(out_path, "wb") as f:
                f.write(png_bytes)
        except Exception:
            # Sample dumping is best-effort; never break the pipeline.
            pass
