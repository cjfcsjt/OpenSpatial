"""
Visualization server for OpenSpatial annotation outputs.

Usage:
    python visualize_server.py --port 8888 --data_dir output/debug

Then open http://<host>:8888 in browser.

Supports two input layouts:
    1. BLINK (default for new runs):
           <data_dir>/<task>.jsonl
           <data_dir>/images/<task>/*.png
    2. Parquet (legacy):
           <data_dir>/**/data.parquet
"""

import argparse
import base64
import io
import json
import os
import glob

import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)
DATA_DIR = "output/debug"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def discover_tasks(data_dir):
    """Scan ``data_dir`` for annotation outputs (BLINK jsonl or parquet).

    Each returned entry carries ``path`` (canonical URL token) plus ``type``
    so the API can dispatch to the right loader.
    """
    results = []

    # BLINK layout: <data_dir>/<task>.jsonl (+ images/<task>/)
    for jsonl_path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        task_name = os.path.splitext(os.path.basename(jsonl_path))[0]
        is_multiview = "multiview" in task_name or task_name.startswith("mmsi_")
        label = f"{'[Multi] ' if is_multiview else '[Single] '}{task_name}"
        results.append({
            "label": label, "path": jsonl_path, "task": task_name,
            "multiview": is_multiview, "type": "blink",
        })

    # Legacy parquet layout: <data_dir>/**/data.parquet
    for pq_path in sorted(glob.glob(os.path.join(data_dir, "**/data.parquet"),
                                    recursive=True)):
        rel = os.path.relpath(pq_path, data_dir)
        parts = rel.split(os.sep)
        task_name = parts[-2] if len(parts) >= 2 else rel
        is_multiview = "multiview" in task_name or task_name.startswith("mmsi_")
        label = f"{'[Multi] ' if is_multiview else '[Single] '}{task_name} (parquet)"
        results.append({
            "label": label, "path": pq_path, "task": task_name,
            "multiview": is_multiview, "type": "parquet",
        })

    return results


def image_from_bytes(data):
    """Convert bytes/dict to PIL Image."""
    if isinstance(data, dict) and "bytes" in data:
        data = data["bytes"]
    if isinstance(data, bytes):
        return Image.open(io.BytesIO(data))
    return None


def pil_to_base64(img, max_w=800):
    """Convert PIL image to base64 data URI, resize if too large."""
    if img is None:
        return ""
    w, h = img.size
    if w > max_w:
        ratio = max_w / w
        img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def load_original_image(image_field):
    """Load original image from path string, bytes, or dict."""
    if isinstance(image_field, str) and os.path.exists(image_field):
        return Image.open(image_field)
    if isinstance(image_field, (bytes, dict)):
        return image_from_bytes(image_field)
    if isinstance(image_field, np.ndarray):
        # multiview: list of image paths
        imgs = []
        for item in image_field:
            if isinstance(item, str) and os.path.exists(item):
                imgs.append(Image.open(item))
        return imgs if imgs else None
    return None


def parse_row(row):
    """Parse a single parquet row into a display-friendly dict.

    Supports both single-turn (2 messages) and multi-turn (4+ messages) conversations.
    Returns a list of (question, answer) turns.
    """
    messages = row.get("messages", [])
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()

    # Parse all turns as (question, answer) pairs
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("from") == "human":
            q = msg.get("value", "")
            a = ""
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if isinstance(next_msg, dict) and next_msg.get("from") == "gpt":
                    a = next_msg.get("value", "")
                    i += 1
            turns.append({"question": q, "answer": a})
        i += 1

    # QA images
    qa_images_raw = row.get("QA_images", None)
    qa_images = []
    if isinstance(qa_images_raw, dict):
        img = image_from_bytes(qa_images_raw)
        if img:
            qa_images.append(img)
    elif isinstance(qa_images_raw, (list, np.ndarray)):
        for item in qa_images_raw:
            img = image_from_bytes(item)
            if img:
                qa_images.append(img)

    tags = row.get("question_tags", [])
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    qtype = row.get("question_types", "")

    return {
        "turns": turns,
        "qa_images": qa_images,
        "tags": tags,
        "question_type": qtype,
    }


def _read_jsonl(path):
    """Lazy-ish jsonl reader: returns a list of records."""
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_blink_record(record, blink_root):
    """Parse a single BLINK jsonl record into the same shape as parse_row."""
    conversations = record.get("conversations", []) or []

    turns = []
    i = 0
    while i < len(conversations):
        msg = conversations[i]
        if isinstance(msg, dict) and msg.get("from") == "human":
            q = msg.get("value", "")
            a = ""
            if i + 1 < len(conversations):
                nxt = conversations[i + 1]
                if isinstance(nxt, dict) and nxt.get("from") == "gpt":
                    a = nxt.get("value", "")
                    i += 1
            turns.append({"question": q, "answer": a})
        i += 1

    qa_images = []
    for rel in record.get("image", []) or []:
        if not isinstance(rel, str):
            continue
        abs_path = rel if os.path.isabs(rel) else os.path.join(blink_root, rel)
        if os.path.exists(abs_path):
            try:
                qa_images.append(Image.open(abs_path))
            except Exception:
                continue

    others = record.get("others") or {}
    tags = others.get("question_tags", [])
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    qtype = others.get("question_types", record.get("output_type", ""))

    return {
        "turns": turns,
        "qa_images": qa_images,
        "tags": tags,
        "question_type": qtype,
    }


# ──────────────────────────────────────────────────────────────────────
# HTML Template
# ──────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenSpatial Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header select { padding: 8px 12px; border-radius: 6px; border: none; font-size: 14px; background: #16213e; color: white; cursor: pointer; min-width: 280px; }
  .header select option { background: #16213e; }
  .header .info { margin-left: auto; font-size: 13px; opacity: 0.8; }
  .nav { display: flex; align-items: center; gap: 8px; margin-left: 16px; }
  .nav button { padding: 6px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.3); background: transparent; color: white; cursor: pointer; font-size: 13px; transition: all 0.2s; }
  .nav button:hover { background: rgba(255,255,255,0.15); }
  .nav button:disabled { opacity: 0.3; cursor: default; }
  .nav span { color: rgba(255,255,255,0.7); font-size: 13px; min-width: 80px; text-align: center; }
  .container { max-width: 1200px; margin: 24px auto; padding: 0 24px; }
  .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; overflow: hidden; }
  .card-header { padding: 14px 20px; background: #f8f9fa; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 10px; }
  .tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .tag-task { background: #e3f2fd; color: #1565c0; }
  .tag-type { background: #f3e5f5; color: #7b1fa2; }
  .card-body { padding: 20px; }
  .images-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .images-row img { border-radius: 8px; border: 1px solid #eee; cursor: pointer; transition: transform 0.2s; max-height: 400px; object-fit: contain; }
  .images-row img:hover { transform: scale(1.02); }
  .qa-block { margin-top: 12px; }
  .qa-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .qa-label.q { color: #1565c0; }
  .qa-label.a { color: #2e7d32; }
  .qa-text { padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
  .qa-text.q { background: #e3f2fd; }
  .qa-text.a { background: #e8f5e9; }
  .turn-divider { border: none; border-top: 1px dashed #ddd; margin: 12px 0; }
  .turn-badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; background: #fff3e0; color: #e65100; margin-left: 6px; }
  .multi-turn-label { font-size: 12px; color: #888; margin-bottom: 4px; }
  /* Lightbox */
  .lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 200; justify-content: center; align-items: center; cursor: zoom-out; }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: 8px; }
  .empty-state { text-align: center; padding: 80px 20px; color: #999; }
  .empty-state h2 { font-size: 24px; margin-bottom: 8px; }
</style>
</head>
<body>

<div class="header">
  <h1>OpenSpatial Visualizer</h1>
  <select id="taskSelect" onchange="loadTask()">
    <option value="">-- Select a task --</option>
    {% for t in tasks %}
    <option value="{{ t.path }}" data-kind="{{ t.type }}" {{ 'selected' if t.path == selected_path else '' }}>{{ t.label }}</option>
    {% endfor %}
  </select>
  <div class="nav">
    <button id="prevBtn" onclick="navigate(-1)" disabled>&larr; Prev</button>
    <span id="pageInfo">-</span>
    <button id="nextBtn" onclick="navigate(1)" disabled>Next &rarr;</button>
  </div>
  <div class="info" id="totalInfo"></div>
</div>

<div class="container" id="content">
  <div class="empty-state">
    <h2>Select a task to visualize</h2>
    <p>Choose an annotation output from the dropdown above</p>
  </div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightboxImg" src="" />
</div>

<script>
const PAGE_SIZE = 10;
let currentPage = 0;
let totalRows = 0;

function loadTask() {
  const sel = document.getElementById('taskSelect');
  const path = sel.value;
  if (!path) return;
  currentPage = 0;
  fetchPage(path, 0);
}

function navigate(delta) {
  const sel = document.getElementById('taskSelect');
  const path = sel.value;
  if (!path) return;
  currentPage += delta;
  fetchPage(path, currentPage);
}

function currentKind() {
  const sel = document.getElementById('taskSelect');
  const opt = sel.options[sel.selectedIndex];
  return opt ? (opt.dataset.kind || '') : '';
}

function fetchPage(path, page) {
  const kind = currentKind();
  fetch(`/api/data?path=${encodeURIComponent(path)}&kind=${encodeURIComponent(kind)}&page=${page}&page_size=${PAGE_SIZE}`)
    .then(r => r.json())
    .then(data => {
      totalRows = data.total;
      currentPage = data.page;
      renderRows(data.rows);
      updateNav();
    });
}

function updateNav() {
  const totalPages = Math.ceil(totalRows / PAGE_SIZE);
  document.getElementById('pageInfo').textContent = totalPages > 0 ? `${currentPage + 1} / ${totalPages}` : '-';
  document.getElementById('prevBtn').disabled = currentPage <= 0;
  document.getElementById('nextBtn').disabled = currentPage >= totalPages - 1;
  document.getElementById('totalInfo').textContent = `${totalRows} rows total`;
}

function renderRows(rows) {
  const container = document.getElementById('content');
  if (rows.length === 0) {
    container.innerHTML = '<div class="empty-state"><h2>No data</h2><p>This task produced no output rows.</p></div>';
    return;
  }
  let html = '';
  rows.forEach((row, idx) => {
    const globalIdx = currentPage * PAGE_SIZE + idx;
    const tagsHtml = (row.tags || []).map(t => `<span class="tag tag-task">${t}</span>`).join(' ');
    const typeHtml = row.question_type ? `<span class="tag tag-type">${row.question_type}</span>` : '';
    const isMultiTurn = row.turns && row.turns.length > 1;
    const turnBadge = isMultiTurn ? `<span class="turn-badge">${row.turns.length} turns</span>` : '';

    let imagesHtml = '';
    if (row.qa_images && row.qa_images.length > 0) {
      const imgTags = row.qa_images.map(src =>
        `<img src="${src}" onclick="openLightbox('${src}')" style="max-width:${row.qa_images.length > 1 ? Math.floor(100/Math.min(row.qa_images.length, 4)) - 2 : 100}%;" />`
      ).join('');
      imagesHtml = `<div class="images-row">${imgTags}</div>`;
    }

    let turnsHtml = '';
    if (row.turns && row.turns.length > 0) {
      row.turns.forEach((turn, tIdx) => {
        const cleanQ = (turn.question || '').replace(/<image>\s*/g, '').trim();
        const turnLabel = isMultiTurn ? `<span class="multi-turn-label">Turn ${tIdx + 1}</span>` : '';
        if (tIdx > 0) turnsHtml += '<hr class="turn-divider">';
        turnsHtml += `
          ${turnLabel}
          <div class="qa-block">
            <div class="qa-label q">Question</div>
            <div class="qa-text q">${escapeHtml(cleanQ)}</div>
          </div>
          <div class="qa-block" style="margin-top: 10px;">
            <div class="qa-label a">Answer</div>
            <div class="qa-text a">${escapeHtml(turn.answer || '')}</div>
          </div>`;
      });
    }

    html += `
    <div class="card">
      <div class="card-header">
        <strong>#${globalIdx + 1}</strong>
        ${tagsHtml} ${typeHtml} ${turnBadge}
      </div>
      <div class="card-body">
        ${imagesHtml}
        ${turnsHtml}
      </div>
    </div>`;
  });
  container.innerHTML = html;
  window.scrollTo(0, 0);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
});

// Auto-load if a task is pre-selected
window.onload = () => {
  const sel = document.getElementById('taskSelect');
  if (sel.value) loadTask();
};
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    tasks = discover_tasks(DATA_DIR)
    selected = request.args.get("task", "")
    return render_template_string(HTML_TEMPLATE, tasks=tasks, selected_path=selected)


@app.route("/api/data")
def api_data():
    path = request.args.get("path", "")
    kind = request.args.get("kind", "")
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 10))

    if not path or not os.path.exists(path):
        return jsonify({"total": 0, "page": 0, "rows": []})

    # Auto-detect when kind is absent (older front-end cache).
    if not kind:
        kind = "blink" if path.endswith(".jsonl") else "parquet"

    if kind == "blink":
        records = _read_jsonl(path)
        total = len(records)
        start = page * page_size
        end = min(start + page_size, total)
        blink_root = os.path.dirname(os.path.abspath(path))

        rows = []
        for i in range(start, end):
            parsed = parse_blink_record(records[i], blink_root)
            img_b64_list = [pil_to_base64(img) for img in parsed["qa_images"]]
            rows.append({
                "turns": parsed["turns"],
                "qa_images": img_b64_list,
                "tags": parsed["tags"] if isinstance(parsed["tags"], list) else [parsed["tags"]],
                "question_type": parsed["question_type"],
            })
        return jsonify({"total": total, "page": page, "rows": rows})

    # Parquet (legacy) path
    df = pd.read_parquet(path)
    total = len(df)
    start = page * page_size
    end = min(start + page_size, total)

    rows = []
    for i in range(start, end):
        row = df.iloc[i]
        parsed = parse_row(row)
        img_b64_list = [pil_to_base64(img) for img in parsed["qa_images"]]
        rows.append({
            "turns": parsed["turns"],
            "qa_images": img_b64_list,
            "tags": parsed["tags"] if isinstance(parsed["tags"], list) else [parsed["tags"]],
            "question_type": parsed["question_type"],
        })

    return jsonify({"total": total, "page": page, "rows": rows})


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSpatial Annotation Visualizer")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--data_dir", type=str, default="output/debug", help="Root directory containing parquet outputs")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    tasks = discover_tasks(DATA_DIR)
    print(f"Found {len(tasks)} task outputs in {DATA_DIR}:")
    for t in tasks:
        print(f"  {t['label']} -> {t['path']}")
    print(f"\nStarting server at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
