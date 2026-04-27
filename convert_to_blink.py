#!/usr/bin/env python3
"""
convert_to_blink.py — Convert OpenSpatial QA parquet files to BLINK format.

Reads annotation parquet files produced by the OpenSpatial pipeline,
extracts embedded images to an images/ folder, and writes a JSONL file
with relative image paths and conversation metadata.

Usage:
    # Convert a single parquet file
    python convert_to_blink.py \
        --input /path/to/03_annotation/base_pipeline_demo_multiview_clockwise/.../data.parquet \
        --output_dir /path/to/blink_output

    # Convert all parquet files under an annotation directory
    python convert_to_blink.py \
        --input_dir /path/to/03_annotation \
        --output_dir /path/to/blink_output

    # Merge all tasks into a single JSONL
    python convert_to_blink.py \
        --input_dir /path/to/03_annotation \
        --output_dir /path/to/blink_output \
        --merge

    # Specify data source name
    python convert_to_blink.py \
        --input_dir /path/to/03_annotation \
        --output_dir /path/to/blink_output \
        --data_source OpenSpatial_ScanNetPP
"""

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage


# ─── Image Extraction ────────────────────────────────────────────────────────

def bytes_dict_to_pil(img_data):
    """Convert a {"bytes": ...} dict or raw bytes to PIL Image."""
    if isinstance(img_data, dict) and img_data.get("bytes"):
        return PILImage.open(io.BytesIO(img_data["bytes"]))
    if isinstance(img_data, bytes):
        return PILImage.open(io.BytesIO(img_data))
    if isinstance(img_data, PILImage.Image):
        return img_data
    return None


def save_image(pil_img, save_path):
    """Save a PIL image to disk as PNG."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_img.save(save_path, format="PNG")


def extract_images_from_row(row, sample_idx, task_name, images_dir):
    """Extract QA_images (processed) from a single row.

    For singleview tasks, QA_images is a single {"bytes": ...} dict.
    For multiview tasks, QA_images is a list of {"bytes": ...} dicts.

    Returns:
        list of relative image paths (relative to output_dir).
    """
    qa_images = row.get("QA_images")
    image_paths = []

    if qa_images is None:
        # Fall back to the original 'image' column
        qa_images = row.get("image")

    if qa_images is None:
        return image_paths

    # Normalize to list
    if isinstance(qa_images, dict):
        qa_images = [qa_images]
    elif not isinstance(qa_images, (list, tuple)):
        # Could be a single bytes object or other format
        qa_images = [qa_images]

    for img_idx, img_data in enumerate(qa_images):
        pil_img = bytes_dict_to_pil(img_data)
        if pil_img is None:
            continue

        # Naming: images/<task_name>/<sample_idx>_view<img_idx>.png
        filename = f"{sample_idx:06d}_view{img_idx}.png"
        rel_path = os.path.join("images", task_name, filename)
        abs_path = os.path.join(images_dir, task_name, filename)

        save_image(pil_img, abs_path)
        image_paths.append(rel_path)

    return image_paths


# ─── Message Parsing ─────────────────────────────────────────────────────────

def parse_messages(messages):
    """Parse OpenSpatial messages format into BLINK conversations.

    OpenSpatial format:
        [{"from": "human", "value": "<image> question..."}, {"from": "gpt", "value": "answer"}]

    BLINK format:
        [{"from": "human", "value": "question..."}, {"from": "gpt", "value": "answer"}]

    Also strips <image> tags from the question text since images are
    referenced by path in the BLINK format.

    Returns:
        (conversations, num_images_in_prompt)
    """
    if messages is None:
        return [], 0

    conversations = []
    num_images = 0

    for msg in messages:
        role = msg.get("from", "")
        value = msg.get("value", "")

        if role == "human":
            # Count and strip <image> tags
            img_count = value.count("<image>")
            num_images += img_count
            # Remove <image> tags and clean up whitespace
            clean_value = re.sub(r'<image>\s*', '', value).strip()
            conversations.append({"from": "human", "value": clean_value})
        else:
            conversations.append({"from": role, "value": value})

    return conversations, num_images


def infer_task_name(parquet_path):
    """Infer task name from parquet file path.

    Expected path pattern:
        .../base_pipeline_demo_multiview_clockwise/annotation_stage/multiview_clockwise/data.parquet

    Returns the task name (e.g., 'multiview_clockwise').
    """
    parts = Path(parquet_path).parts
    # Look for 'annotation_stage' in path and take the next part
    for i, part in enumerate(parts):
        if part == "annotation_stage" and i + 1 < len(parts):
            return parts[i + 1]

    # Fallback: look for 'base_pipeline_demo_' prefix
    for part in parts:
        if part.startswith("base_pipeline_demo_"):
            return part.replace("base_pipeline_demo_", "")

    # Last resort: use parent directory name
    return Path(parquet_path).parent.name


def infer_output_type(conversations):
    """Infer output type (MCQ or open-ended) from conversations."""
    if not conversations:
        return "open"

    human_text = ""
    for msg in conversations:
        if msg.get("from") == "human":
            human_text += msg.get("value", "")

    # Check for MCQ patterns: A. / B. / C. / D. or (A) / (B) etc.
    mcq_pattern = re.compile(r'\b[A-D]\.\s|\([A-D]\)')
    if mcq_pattern.search(human_text):
        return "MCQ"
    return "open"


def classify_task_type(task_name):
    """Classify task into singleview or multiview category."""
    multiview_prefixes = ["multiview_", "mmsi_"]
    for prefix in multiview_prefixes:
        if task_name.startswith(prefix):
            return "multiview"
    return "singleview"


# ─── Core Conversion ─────────────────────────────────────────────────────────

def convert_parquet_to_blink(parquet_path, output_dir, data_source="OpenSpatial",
                             task_name_override=None):
    """Convert a single QA parquet file to BLINK format.

    Args:
        parquet_path: Path to the annotation parquet file.
        output_dir: Root output directory.
        data_source: Data source name for BLINK metadata.
        task_name_override: Override inferred task name.

    Returns:
        (list of BLINK records, task_name, stats_dict)
    """
    if not os.path.exists(parquet_path):
        print(f"  ⚠️  File not found: {parquet_path}")
        return [], "", {}

    print(f"  📂 Reading: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    print(f"  📊 Rows: {len(df)}")

    task_name = task_name_override or infer_task_name(parquet_path)
    task_type = classify_task_type(task_name)
    images_dir = os.path.join(output_dir, "images")

    records = []
    stats = {"total": len(df), "converted": 0, "skipped": 0, "image_errors": 0}

    for idx in range(len(df)):
        row = df.iloc[idx].to_dict()

        try:
            # Parse messages
            messages = row.get("messages")
            if messages is None:
                stats["skipped"] += 1
                continue

            conversations, num_images_in_prompt = parse_messages(messages)
            if not conversations:
                stats["skipped"] += 1
                continue

            # Extract and save images
            image_paths = extract_images_from_row(row, idx, task_name, images_dir)
            if not image_paths:
                stats["image_errors"] += 1
                # Still include the record but with empty image list
                pass

            # Infer output type
            output_type = infer_output_type(conversations)

            # Extract question tags and types
            question_tags = row.get("question_tags", [])
            question_types = row.get("question_types", "")

            # Build BLINK record
            record = {
                "id": f"{data_source}_{task_name}_{idx:06d}",
                "image": image_paths,
                "video": [],
                "conversations": conversations,
                "task": task_name,
                "input_type": "image",
                "output_type": output_type,
                "data_source": data_source,
                "sub_task": task_type,
                "others": {
                    "question_tags": question_tags,
                    "question_types": question_types,
                }
            }

            records.append(record)
            stats["converted"] += 1

        except Exception as e:
            print(f"  ⚠️  Error processing row {idx}: {e}")
            stats["skipped"] += 1
            continue

    print(f"  ✅ Converted: {stats['converted']}/{stats['total']} "
          f"(skipped: {stats['skipped']}, image_errors: {stats['image_errors']})")

    return records, task_name, stats


def find_annotation_parquets(input_dir):
    """Find all data.parquet files under an annotation output directory.

    Returns:
        list of (parquet_path, inferred_task_name)
    """
    results = []
    input_path = Path(input_dir)

    for parquet_file in sorted(input_path.rglob("data.parquet")):
        task_name = infer_task_name(str(parquet_file))
        results.append((str(parquet_file), task_name))

    # Also find batch parts: data_part_0.parquet, data_part_1.parquet, etc.
    for parquet_file in sorted(input_path.rglob("data_part_*.parquet")):
        task_name = infer_task_name(str(parquet_file))
        results.append((str(parquet_file), task_name))

    return results


def write_jsonl(records, output_path):
    """Write BLINK records to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    print(f"  📁 Written: {output_path} ({len(records)} records)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenSpatial QA parquet files to BLINK format"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i",
        help="Path to a single annotation parquet file"
    )
    group.add_argument(
        "--input_dir", "-d",
        help="Path to annotation output directory (scans for all data.parquet files)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Output directory for BLINK format files"
    )
    parser.add_argument(
        "--data_source",
        default="OpenSpatial",
        help="Data source name for BLINK metadata (default: OpenSpatial)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all tasks into a single JSONL file (default: one JSONL per task)"
    )
    parser.add_argument(
        "--task_name",
        default=None,
        help="Override task name (only for single --input mode)"
    )

    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_records = []
    all_stats = {}

    if args.input:
        # Single file mode
        parquet_path = os.path.abspath(args.input)
        records, task_name, stats = convert_parquet_to_blink(
            parquet_path, output_dir,
            data_source=args.data_source,
            task_name_override=args.task_name,
        )
        all_records.extend(records)
        all_stats[task_name] = stats

        if not args.merge:
            jsonl_path = os.path.join(output_dir, f"{task_name}.jsonl")
            write_jsonl(records, jsonl_path)

    elif args.input_dir:
        # Directory scan mode
        input_dir = os.path.abspath(args.input_dir)
        parquet_files = find_annotation_parquets(input_dir)

        if not parquet_files:
            print(f"❌ No annotation parquet files found under: {input_dir}")
            sys.exit(1)

        print(f"📋 Found {len(parquet_files)} parquet file(s):\n")
        for pf, tn in parquet_files:
            print(f"   {tn:40s} → {pf}")
        print()

        for parquet_path, task_name in parquet_files:
            print(f"{'=' * 60}")
            print(f"🔄 Processing: {task_name}")
            print(f"{'=' * 60}")

            records, task_name, stats = convert_parquet_to_blink(
                parquet_path, output_dir,
                data_source=args.data_source,
            )
            all_records.extend(records)

            # Accumulate stats per task
            if task_name in all_stats:
                for k in stats:
                    if isinstance(stats[k], int):
                        all_stats[task_name][k] += stats[k]
            else:
                all_stats[task_name] = stats

            if not args.merge:
                jsonl_path = os.path.join(output_dir, f"{task_name}.jsonl")
                write_jsonl(records, jsonl_path)

    # Write merged file if requested
    if args.merge and all_records:
        merged_path = os.path.join(output_dir, f"{args.data_source}_all.jsonl")
        write_jsonl(all_records, merged_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"📊 Conversion Summary")
    print(f"{'=' * 60}")
    total_converted = 0
    total_skipped = 0
    total_img_err = 0
    for task_name, stats in sorted(all_stats.items()):
        converted = stats.get("converted", 0)
        skipped = stats.get("skipped", 0)
        img_err = stats.get("image_errors", 0)
        total_converted += converted
        total_skipped += skipped
        total_img_err += img_err
        print(f"  {task_name:40s}  converted={converted:>6d}  "
              f"skipped={skipped:>4d}  img_err={img_err:>4d}")

    print(f"  {'─' * 70}")
    print(f"  {'TOTAL':40s}  converted={total_converted:>6d}  "
          f"skipped={total_skipped:>4d}  img_err={total_img_err:>4d}")
    print(f"\n📁 Output directory: {output_dir}")
    if args.merge:
        print(f"📄 Merged JSONL: {os.path.join(output_dir, f'{args.data_source}_all.jsonl')}")
    print(f"🖼️  Images directory: {os.path.join(output_dir, 'images')}")


if __name__ == "__main__":
    main()
