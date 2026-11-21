"""评估脚本：按 CCPD 文件名真值计算检测与识别指标，并按场景分组统计。

真值来源：CCPD 文件名自带的边界框、车牌号码、倾斜角、亮度、模糊度。

用法:
    # 1. 先跑识别得到 CSV（不写结果图，速度更快）
    python scripts/run_integrated.py --source data/test_images --no-save \
        --csv samples/results/ccpd300.csv
    # 2. 评估
    python scripts/evaluate.py --results samples/results/ccpd300.csv \
        --output samples/results/eval_report.md
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_ccpd import parse_plate  # noqa: E402

IOU_THRESHOLD = 0.5


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_gt(filename: str) -> dict:
    """解析 CCPD 文件名的真值；非 CCPD 命名抛 ValueError。"""
    parts = Path(filename).stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"不是 CCPD 文件名：{filename}")
    bbox = tuple(int(v) for v in parts[2].replace("_", "&").split("&"))
    tilt = tuple(int(v) for v in parts[1].split("_"))
    return {
        "plate": parse_plate(filename),
        "bbox": bbox,
        "tilt": tilt,
        "brightness": int(parts[-2]),
        "blur": int(parts[-1]),
    }


def bucket_blur(v: int) -> str:
    if v <= 2:
        return "清晰（blur≤2）"
    if v <= 5:
        return "轻微模糊（3-5）"
    return "明显模糊（≥6）"


def bucket_tilt(tilt) -> str:
    h, v = tilt
    d = max(abs(h - 90), abs(v - 90))
    if d <= 5:
        return "基本水平（偏移≤5°）"
    if d <= 15:
        return "轻微倾斜（6-15°）"
    return "明显倾斜（>15°）"


def bucket_brightness(v: int) -> str:
    if v < 100:
        return "偏暗（<100）"
    if v <= 180:
        return "正常（100-180）"
    return "偏亮（>180）"


def load_results(csv_path: Path) -> dict:
    """读取识别 CSV，按文件名分组：preds 列表 + 单图耗时。"""
    groups = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = Path(row["image"]).name
            entry = groups.setdefault(name, {"preds": [], "elapsed_ms": 0.0})
            if row.get("elapsed_ms"):
                entry["elapsed_ms"] = float(row["elapsed_ms"])
            if row.get("plate"):
                entry["preds"].append(
                    {
                        "plate": row["plate"],
                        "bbox": (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                    }
                )
    return groups


def evaluate(groups: dict, iou_threshold: float = IOU_THRESHOLD) -> dict:
    stats = {
        "images": 0, "skipped": 0, "detected": 0,
        "gt_plates": 0, "tp": 0, "fp": 0, "fn": 0,
        "plate_correct": 0, "char_acc_sum": 0.0,
        "elapsed": [],
    }
    scenes = defaultdict(lambda: {"images": 0, "detected": 0, "correct": 0})
    failures = []

    for name in sorted(groups):
        try:
            gt = parse_gt(name)
        except ValueError:
            stats["skipped"] += 1
            continue

        entry = groups[name]
        preds = entry["preds"]
        stats["images"] += 1
        stats["gt_plates"] += 1
        stats["elapsed"].append(entry["elapsed_ms"])

        best_iou, best_pred = 0.0, None
        for pred in preds:
            iou = _iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_pred = iou, pred
        matched = best_pred is not None and best_iou >= iou_threshold

        detected = len(preds) > 0
        stats["detected"] += int(detected)
        if matched:
            stats["tp"] += 1
            stats["fp"] += max(0, len(preds) - 1)
        else:
            stats["fn"] += 1
            stats["fp"] += len(preds)

        correct = matched and best_pred["plate"] == gt["plate"]
        if correct:
            stats["plate_correct"] += 1
            stats["char_acc_sum"] += 1.0
        else:
            if matched:
                base = best_pred["plate"]
                stats["char_acc_sum"] += 1 - levenshtein(base, gt["plate"]) / len(gt["plate"])
                failures.append((name, gt["plate"], base, "号码误识"))
            elif detected:
                failures.append((name, gt["plate"], best_pred["plate"] if best_pred else preds[0]["plate"], "框未对齐"))
            else:
                failures.append((name, gt["plate"], "-", "漏检"))

        for dimension, value in (
            ("模糊度", bucket_blur(gt["blur"])),
            ("倾斜", bucket_tilt(gt["tilt"])),
            ("亮度", bucket_brightness(gt["brightness"])),
        ):
            scene = scenes[(dimension, value)]
            scene["images"] += 1
            scene["detected"] += int(detected)
            scene["correct"] += int(correct)

    return {"stats": stats, "scenes": scenes, "failures": failures}


def _pct(a, b) -> str:
    return f"{a / b * 100:.1f}%" if b else "-"


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def render_report(result: dict, csv_path: str) -> str:
    s = result["stats"]
    lines = []
    lines.append("# 方案一（HyperLPR3 集成式）评估报告\n")
    lines.append(f"- 结果文件：`{csv_path}`")
    lines.append(f"- 有效图片：{s['images']} 张（跳过非 CCPD 命名 {s['skipped']} 张）")
    lines.append(f"- 有检出：{s['detected']} 张（{_pct(s['detected'], s['images'])}）")
    lines.append(f"- 车牌真值：{s['gt_plates']} 个")
    lines.append(
        f"- 检测 TP / FP / FN：{s['tp']} / {s['fp']} / {s['fn']}，"
        f"精确率 {_pct(s['tp'], s['tp'] + s['fp'])}，召回率 {_pct(s['tp'], s['tp'] + s['fn'])}"
    )
    lines.append(f"- **整牌正确率（图片级）：{_pct(s['plate_correct'], s['images'])}**（{s['plate_correct']}/{s['images']}）")
    lines.append(f"- 字符准确率：{_pct(s['char_acc_sum'], s['images'])}")
    lines.append(
        f"- 平均耗时：{_mean(s['elapsed']) / 1000:.2f} s/张"
        f"（P50 {_percentile(s['elapsed'], 0.5) / 1000:.2f} s，"
        f"P95 {_percentile(s['elapsed'], 0.95) / 1000:.2f} s）"
    )

    lines.append("\n## 分场景结果\n")
    lines.append("| 维度 | 场景 | 图片数 | 检出率 | 整牌正确率 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for (dimension, value), scene in sorted(result["scenes"].items()):
        lines.append(
            f"| {dimension} | {value} | {scene['images']} | "
            f"{_pct(scene['detected'], scene['images'])} | "
            f"{_pct(scene['correct'], scene['images'])} |"
        )

    if result["failures"]:
        lines.append("\n## 失败案例（前 20 条）\n")
        lines.append("| 图片 | 真值 | 预测 | 类型 |")
        lines.append("| --- | --- | --- | --- |")
        for name, gt, pred, kind in result["failures"][:20]:
            lines.append(f"| {name} | {gt} | {pred} | {kind} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="按 CCPD 真值评估识别结果")
    parser.add_argument("--results", required=True, help="run_integrated.py 输出的 CSV")
    parser.add_argument("--output", default=None, help="评估报告（markdown）输出路径")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="检测匹配 IoU 阈值（默认 0.5）")
    args = parser.parse_args()

    groups = load_results(Path(args.results))
    result = evaluate(groups, iou_threshold=args.iou)
    report = render_report(result, args.results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"报告已保存：{output_path}")
    print(report)


if __name__ == "__main__":
    main()
