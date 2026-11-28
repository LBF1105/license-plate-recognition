#!/usr/bin/env python3
"""Score the self-shot Xiang test set (data/test_images_xiang/).

Reads labels.csv (filename, true_plate, scene, type) and the two result
CSVs produced by scripts/run_integrated.py and scripts/run_two_stage.py,
then prints a per-image table and per-scheme exact-match accuracy.

A photo may contain background plates besides the target one, so results
are grouped by image: an image counts as correct for a scheme when ANY
detection in that image equals the ground-truth plate. Every detection is
listed in the table, so extra (background) detections stay visible instead
of being silently dropped.

Usage (from the repo root):
    # 1. run both schemes on the folder (integrated WITHOUT --tile)
    python scripts/run_integrated.py --source data/test_images_xiang \
        --csv samples/results/integrated_xiang.csv
    python scripts/run_two_stage.py --source data/test_images_xiang \
        --csv samples/results/two_stage_xiang.csv
    # 2. score
    python scripts/score_xiang.py
"""

import argparse
import csv
import sys
from pathlib import Path

IMAGE_COLS = ("image", "filename", "file", "图片", "文件名")
PLATE_COLS = ("plate", "prediction", "plate_number", "车牌", "识别结果")
TRUTH_COLS = ("true_plate", "plate", "ground_truth", "真值", "真值号码")


def norm(plate: str) -> str:
    """Normalize a plate string for comparison: drop separators/spaces."""
    return (
        plate.strip()
        .replace("·", "")
        .replace(".", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("_", "")
        .upper()
    )


def find_col(fieldnames, candidates, kind, path):
    lower = {f.strip().lower(): f for f in fieldnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    print(f"在 {path} 中找不到{kind}列，实际表头：{fieldnames}")
    sys.exit(2)


def load_labels(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fn_col = find_col(reader.fieldnames, ("filename", "image", "file", "文件名"), "文件名", path)
        truth_col = find_col(reader.fieldnames, TRUTH_COLS, "真值", path)
        scene_col = next((c for c in reader.fieldnames if c.strip() in ("scene", "场景")), None)
        rows = []
        for row in reader:
            name = (row.get(fn_col) or "").strip()
            truth = (row.get(truth_col) or "").strip()
            if not name or not truth:
                continue
            rows.append(
                {
                    "file": Path(name).name,
                    "truth": truth,
                    "scene": (row.get(scene_col) or "").strip() if scene_col else "",
                }
            )
    if not rows:
        print(f"{path} 中没有有效的标签行")
        sys.exit(2)
    return rows


def load_results(path: Path):
    """Group detections by image basename: {basename: [plate, ...]}."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        img_col = find_col(reader.fieldnames, IMAGE_COLS, "图片", path)
        plate_col = find_col(reader.fieldnames, PLATE_COLS, "车牌", path)
        grouped = {}
        for row in reader:
            name = Path((row.get(img_col) or "").strip()).name
            plate = (row.get(plate_col) or "").strip()
            if not name:
                continue
            grouped.setdefault(name, [])
            if plate:
                grouped[name].append(plate)
    return grouped


def mark(hit):
    if hit is None:
        return "-"
    return "✓" if hit else "✗"


def main() -> None:
    parser = argparse.ArgumentParser(description="自拍湘牌测试集打分（按图分组计分）")
    parser.add_argument("--labels", default="data/test_images_xiang/labels.csv")
    parser.add_argument("--integrated", default="samples/results/integrated_xiang.csv")
    parser.add_argument("--two-stage", default="samples/results/two_stage_xiang.csv")
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    res_a = load_results(Path(args.integrated))
    res_b = load_results(Path(args.two_stage))
    if res_a is None and res_b is None:
        print("两个结果文件都不存在，请先运行两个方案：")
        print("  python scripts/run_integrated.py --source data/test_images_xiang --csv samples/results/integrated_xiang.csv")
        print("  python scripts/run_two_stage.py --source data/test_images_xiang --csv samples/results/two_stage_xiang.csv")
        sys.exit(1)

    print(f"{'文件':<14} {'真值':<10} {'场景':<6} {'方案一检出':<28} {'方案一':<4} {'方案二检出':<28} {'方案二'}")
    correct_a = correct_b = 0
    total_a = total_b = 0
    for item in labels:
        truth = norm(item["truth"])
        cells = []
        for res in (res_a, res_b):
            if res is None:
                cells.append(("（未运行）", None))
                continue
            plates = res.get(item["file"], [])
            hit = truth in (norm(p) for p in plates)
            cells.append(("; ".join(plates) if plates else "（无检出）", hit))
        if res_a is not None:
            total_a += 1
            correct_a += bool(cells[0][1])
        if res_b is not None:
            total_b += 1
            correct_b += bool(cells[1][1])
        print(
            f"{item['file']:<14} {item['truth']:<10} {item['scene']:<6} "
            f"{cells[0][0]:<28} {mark(cells[0][1]):<4} {cells[1][0]:<28} {mark(cells[1][1])}"
        )
    print()
    if res_a is not None:
        print(f"方案一（HyperLPR3 集成式）整牌正确率：{correct_a}/{total_a} = {correct_a / total_a:.1%}")
    if res_b is not None:
        print(f"方案二（YOLOv8 + LPRNet 分步式）整牌正确率：{correct_b}/{total_b} = {correct_b / total_b:.1%}")


if __name__ == "__main__":
    main()
