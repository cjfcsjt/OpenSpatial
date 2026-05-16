#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量解压指定目录下的全部归档文件，并显示进度。

默认行为：
  - 递归扫描 --root 下的 .zip / .tar / .tar.gz / .tgz / .tar.bz2 / .tbz2 / .tar.xz / .txz 文件
  - 每个归档解压到它所在目录下的同名子目录（例如 41254925.zip -> 41254925/）
  - 解压成功后保留原归档（如需删除，加 --delete-archive）
  - 已经解压过（目标目录存在且非空）的归档默认跳过；--force 可强制重解
  - 显示两级进度条：归档级（总共 N 个归档）+ 单个归档内部（成员级）

用法：
  python extract_arkitscenes.py \
      --root /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/3dod/Training

  # 并行（IO/CPU 都比较快时更省时间）
  python extract_arkitscenes.py --root /path/to/Training --workers 8

  # 强制重新解压，并在成功后删除原归档
  python extract_arkitscenes.py --root /path/to/Training --force --delete-archive

# 先装个进度条库（没装也能跑，只是没有漂亮的 tqdm 进度条）
pip install tqdm

# 基本用法：单线程，带两级进度条
python extract_arkitscenes.py \
    --root /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/3dod/Training

# 先 dry-run 看一下会处理哪些文件（强烈建议第一次先这样跑一遍）
python extract_arkitscenes.py \
    --root /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/3dod/Training \
    --dry-run

# 并行解压（IO 好的机器能快不少；并行模式下不再显示"单个归档内部进度"以免多条进度条互相打架）
python extract_arkitscenes.py \
    --root /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/3dod/Training \
    --workers 8

# 强制重解（目标目录非空也要覆盖），并在解压成功后删除原归档以省磁盘
python scripts/extract_arkitscenes.py \
    --root /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/3dod/Training \
    --force --delete-archive
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # 降级到简单打印
    tqdm = None  # type: ignore


ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz", ".tgz",
    ".tar.bz2", ".tbz2",
    ".tar.xz", ".txz",
)


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suf) for suf in ARCHIVE_SUFFIXES)


def strip_archive_suffix(name: str) -> str:
    low = name.lower()
    for suf in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if low.endswith(suf):
            return name[: -len(suf)]
    return name


def target_dir_for(archive: Path) -> Path:
    """归档解压到与归档同目录下的同名子目录。"""
    return archive.parent / strip_archive_suffix(archive.name)


def find_archives(root: Path) -> List[Path]:
    archives: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_archive(p):
                archives.append(p)
    archives.sort()
    return archives


def dir_nonempty(p: Path) -> bool:
    if not p.is_dir():
        return False
    try:
        next(p.iterdir())
        return True
    except StopIteration:
        return False


def extract_zip(archive: Path, out_dir: Path, show_inner: bool) -> Tuple[int, int]:
    """返回 (成员数, 总字节数)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_members = 0
    total_bytes = 0
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        n_members = len(infos)
        iterator = infos
        if show_inner and tqdm is not None:
            iterator = tqdm(
                infos,
                desc=f"  · {archive.name}",
                unit="file",
                leave=False,
                dynamic_ncols=True,
            )
        for info in iterator:
            zf.extract(info, out_dir)
            total_bytes += info.file_size
    return n_members, total_bytes


def extract_tar(archive: Path, out_dir: Path, show_inner: bool) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # tarfile 根据后缀/魔数自动选择压缩算法
    n_members = 0
    total_bytes = 0
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        n_members = len(members)
        iterator = members
        if show_inner and tqdm is not None:
            iterator = tqdm(
                members,
                desc=f"  · {archive.name}",
                unit="file",
                leave=False,
                dynamic_ncols=True,
            )
        for m in iterator:
            tf.extract(m, out_dir)
            if m.isfile():
                total_bytes += m.size
    return n_members, total_bytes


def extract_one(
    archive: Path,
    force: bool,
    delete_archive: bool,
    show_inner: bool,
) -> Tuple[Path, str, str]:
    """返回 (archive, status, message)。status ∈ {ok, skipped, error}。"""
    out_dir = target_dir_for(archive)
    if not force and dir_nonempty(out_dir):
        return archive, "skipped", f"already extracted -> {out_dir}"
    try:
        low = archive.name.lower()
        t0 = time.time()
        if low.endswith(".zip"):
            n, nbytes = extract_zip(archive, out_dir, show_inner=show_inner)
        else:
            n, nbytes = extract_tar(archive, out_dir, show_inner=show_inner)
        dt = time.time() - t0
        if delete_archive:
            try:
                archive.unlink()
            except OSError as e:
                return archive, "ok", (
                    f"{n} files, {nbytes/1e6:.1f} MB, {dt:.1f}s -> {out_dir}; "
                    f"but failed to delete archive: {e}"
                )
        return (
            archive,
            "ok",
            f"{n} files, {nbytes/1e6:.1f} MB, {dt:.1f}s -> {out_dir}",
        )
    except Exception as e:  # noqa: BLE001
        return archive, "error", f"{type(e).__name__}: {e}"


def human_size(path: Path) -> str:
    try:
        sz = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if sz < 1024:
            return f"{sz:.1f}{unit}"
        sz /= 1024
    return f"{sz:.1f}PB"


def main() -> int:
    ap = argparse.ArgumentParser(description="Recursively extract all archives under a directory.")
    ap.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Directory to scan. e.g. /path/to/ARKitScenes/3dod/Training",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers. >1 enables ThreadPool extraction (default: 1).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if target dir already exists and is non-empty.",
    )
    ap.add_argument(
        "--delete-archive",
        action="store_true",
        help="Remove the original archive file after a successful extraction.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the archives that would be processed, then exit.",
    )
    args = ap.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[FAIL] not a directory: {root}", file=sys.stderr)
        return 2

    print(f"[scan ] root = {root}")
    archives = find_archives(root)
    print(f"[scan ] found {len(archives)} archive(s)")
    if not archives:
        return 0

    if args.dry_run:
        for a in archives:
            print(f"  - {a}  ({human_size(a)})  -> {target_dir_for(a)}")
        return 0

    n_ok = n_skip = n_err = 0
    errors: List[Tuple[Path, str]] = []

    # 并行时不显示成员级进度条（会互相打架）
    show_inner = (args.workers <= 1) and (tqdm is not None)

    if args.workers <= 1:
        iterator = archives
        if tqdm is not None:
            iterator = tqdm(
                archives,
                desc="Archives",
                unit="arc",
                dynamic_ncols=True,
            )
        for a in iterator:
            # 在进度条上显示当前文件名
            if tqdm is not None and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(f"{a.name} ({human_size(a)})")
            _, status, msg = extract_one(
                a,
                force=args.force,
                delete_archive=args.delete_archive,
                show_inner=show_inner,
            )
            if status == "ok":
                n_ok += 1
                tqdm.write(f"[ok   ] {a}  ::  {msg}") if tqdm else print(f"[ok   ] {a}  ::  {msg}")
            elif status == "skipped":
                n_skip += 1
                tqdm.write(f"[skip ] {a}  ::  {msg}") if tqdm else print(f"[skip ] {a}  ::  {msg}")
            else:
                n_err += 1
                errors.append((a, msg))
                tqdm.write(f"[ERR  ] {a}  ::  {msg}") if tqdm else print(f"[ERR  ] {a}  ::  {msg}", file=sys.stderr)
    else:
        print(f"[run  ] using {args.workers} workers (inner progress bars disabled)")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    extract_one,
                    a,
                    args.force,
                    args.delete_archive,
                    False,
                ): a
                for a in archives
            }
            pbar = tqdm(total=len(futures), desc="Archives", unit="arc", dynamic_ncols=True) if tqdm else None
            for fut in as_completed(futures):
                a = futures[fut]
                try:
                    _, status, msg = fut.result()
                except Exception as e:  # noqa: BLE001
                    status, msg = "error", f"{type(e).__name__}: {e}"
                if status == "ok":
                    n_ok += 1
                    line = f"[ok   ] {a}  ::  {msg}"
                elif status == "skipped":
                    n_skip += 1
                    line = f"[skip ] {a}  ::  {msg}"
                else:
                    n_err += 1
                    errors.append((a, msg))
                    line = f"[ERR  ] {a}  ::  {msg}"
                (tqdm.write(line) if pbar is not None else print(line))
                if pbar is not None:
                    pbar.set_postfix_str(f"ok={n_ok} skip={n_skip} err={n_err}")
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

    print()
    print(f"[done ] ok={n_ok}  skipped={n_skip}  error={n_err}  total={len(archives)}")
    if errors:
        print("[done ] failures:")
        for a, msg in errors:
            print(f"   - {a}  ::  {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
