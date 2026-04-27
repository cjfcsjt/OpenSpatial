<p align="center">
  <img src="assets/logo.png" alt="OpenSpatial Logo" width="300">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.07296"><img src="https://img.shields.io/badge/arXiv-2604.07296-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

**OpenSpatial** is an open-source 3D spatial understanding data engine engineered for **high quality**, **extensive scalability**, **broad task diversity**, and **optimized efficiency**. 

By bridging the gap between massive 2D web data and complex 3D spatial reasoning, OpenSpatial provides a comprehensive suite for the next generation of Embodied AI and World Models.

---

<p align="center">
  <img src="assets/teaser.png" alt="OpenSpatial Teaser" width="800">
  <br>
  <em>OpenSpatial Pipeline: From 2D Web Data to 3D Spatial Understanding</em>
</p>

---


## 🔥 News
- **[2026.04.15]** 🎉 We have released the open-source subset of the **OpenSpatial-3M** dataset! Check it out on [Hugging Face](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial).
- **[2026.04.08]** 🎉 The **OpenSpatial 3D data engine** is now officially open-sourced.

## 🚀 Key Features

* **Web Data 3D Lifting**: Advanced pipelines to transform large-scale 2D web imagery into geometrically consistent 3D representations.
* **Diverse Data Generation**: Automated engine for creating rich spatial understanding datasets, covering various environments and object-level details.
* **Multi-Task Integration**: Support for a wide range of tasks including 3D grounding, spatial reasoning, and scene captioning.
* **Comprehensive Evaluation**: Built-in benchmarking suite to evaluate spatial understanding capabilities across different model architectures.
* **High Efficiency**: Optimized for large-scale data processing with scalable distributed computing support.

## 📊 Dataset

The **OpenSpatial-3M** dataset is now available on Hugging Face. It contains 3 million high-fidelity samples designed to enhance 3D spatial reasoning in large multi-modal models.

* **Repository**: [jdopensource/JoyAI-Image-OpenSpatial](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial)

## 📖 Documentation

| Document | Description |
|---|---|
| [Quick Start](assets/quick_start.md) | Data preparation, config structure, annotation pipeline usage, and running tasks end-to-end |
| [Development Guide](assets/development_guide.md) | Adding new annotation tasks, pipeline stages, prompt templates, dataset preprocessors, and internal architecture reference |

## 🗺️ Cognitive Map & Benchmark-Style Tasks

OpenSpatial optionally attaches a **question-related cognitive map** to every
generated QA. A cognitive map is a 10×10 bird's-eye-view grid recording the
cameras (position + yaw) and objects (position + size + yaw) that participate
in the QA, plus a PNG visualization with the question/answer overlay.

Enable it per task by adding the following block to any `config/annotation/*.yaml`:

```yaml
cognitive_map:
  enable: true               # turn the feature on (off by default for back-compat)
  enable_visualization: true # render PNGs alongside the JSON map
  dump_samples: true         # dump the first N PNGs to output_dir/cognitive_map_samples/
  dump_sample_count: 20
  grid_size: 10
  padding_ratio: 0.10
```

The following benchmark-style multiview tasks are available out of the box
(see their dedicated demo YAMLs under `config/annotation/`):

| Task | Benchmark style | YAML |
|---|---|---|
| Relative Distance | all-angles | `demo_multiview_relative_distance.yaml` |
| Clockwise (Yes/No) | BLINK | `demo_multiview_clockwise.yaml` |
| Camera Movement Direction | VSI-Bench | `demo_multiview_camera_movement.yaml` |
| Camera–Camera | MMSI-Bench | `demo_mmsi_camera_camera.yaml` |
| Camera–Object | MMSI-Bench | `demo_mmsi_camera_object.yaml` |
| Object–Object (world-frame) | MMSI-Bench | `demo_mmsi_object_object.yaml` |
| Camera Motion (composite) | MMSI-Bench | `demo_mmsi_camera_motion.yaml` |
| BEV Pose Estimation | all-angles | `demo_multiview_bev_pose_estimation.yaml` |
| Manipulation Viewpoint | all-angles (approx.) | `demo_multiview_manipulation_view.yaml` |


## 📅 Roadmap & To-Do List

- [x] **3D Data Engine**: Open-source the core 3D spatial understanding data engine.
- [x] **OpenSpatial-3M Dataset Release**: Publicly release the large-scale 3M spatial understanding dataset. [[HF Link]](https://huggingface.co/datasets/jdopensource/JoyAI-Image-OpenSpatial)
- [ ] **Model Release**: Release the trained spatial understanding model.
- [ ] **Evaluation Suite**: Open-source the comprehensive evaluation code for spatial tasks.
- [ ] **3D Lifting Module**: Integrate the core engine for lifting 2D web data to 3D representations.
- [ ] **More Tasks**: Extend support for more spatial understanding task types.

## 📄 Citation

If you find OpenSpatial useful for your research, please consider citing our paper:

```bibtex
@article{openspatial2025,
  title={OpenSpatial: An Open-Source 3D Spatial Understanding Data Engine},
  journal={arXiv preprint arXiv:2604.07296},
  year={2025}
}
