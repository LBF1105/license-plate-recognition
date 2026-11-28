"""对比实验：关闭 / 开启「自动拉正」，量化它到底带来多少收益。

做法：同一张图、同一个检测器，分别走两条链路
    baseline : YOLO → 裁剪 → LPRNet
    deskew   : YOLO → 裁剪 → 自动拉正 → LPRNet
再按 CCPD 文件名的真值（号码、倾角）统计整牌正确率。

注意：真值只在“评估”阶段读取，识别链路本身完全不看文件名，
自动拉正只用裁剪图自身的像素信息。

用法:
    python scripts/compare_deskew.py --source data/test_images \
        --output samples/results/deskew_report.md
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from evaluate import IOU_THRESHOLD, _iou, bucket_tilt, levenshtein, parse_gt  # noqa: E402
from plate_recognition.common.image_io import list_images, load_image  # noqa: E402
from plate_recognition.two_stage.deskewer import PlateDeskewer  # noqa: E402
from plate_recognition.two_stage.detector import PlateDetector  # noqa: E402
from plate_recognition.two_stage.pipeline import TwoStagePipeline  # noqa: E402
from plate_recognition.two_stage.recognizer import PlateRecognizer  # noqa: E402

TILT_ORDER = ["基本水平（偏移≤5°）", "轻微倾斜（6-15°）", "明显倾斜（>15°）"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def match(preds, gt_bbox):
    """按 IoU 找与真值框最匹配的预测。"""
    best_iou, best = 0.0, None
    for pred in preds:
        iou = _iou(pred["bbox"], gt_bbox)
        if iou > best_iou:
            best_iou, best = iou, pred
    return best_iou, best


def evaluate_pipeline(pipeline, images):
    """跑一条链路，返回 {图片名: 记录}。"""
    records = {}
    for path in images:
        image = load_image(path)
        start = time.perf_counter()
        results = pipeline.run(image)
        records[Path(path).name] = {
            "results": results,
            "elapsed_ms": (time.perf_counter() - start) * 1000,
        }
    return records


def summarize(records):
    stats = {
        "images": 0,
        "detected": 0,
        "correct": 0,
        "char_acc": 0.0,
        "elapsed": [],
        "applied": 0,
        "methods": defaultdict(int),
        "angles": [],
    }
    per_image = {}
    for name, entry in records.items():
        try:
            gt = parse_gt(name)
        except ValueError:
            continue
        stats["images"] += 1
        stats["elapsed"].append(entry["elapsed_ms"])
        preds = entry["results"]
        best_iou, best = match(preds, gt["bbox"])
        matched = best is not None and best_iou >= IOU_THRESHOLD
        stats["detected"] += int(bool(preds))
        if matched:
            correct = best["plate"] == gt["plate"]
            if correct:
                stats["correct"] += 1
                stats["char_acc"] += 1.0
            else:
                stats["char_acc"] += 1 - levenshtein(best["plate"], gt["plate"]) / len(gt["plate"])
            per_image[name] = {
                "correct": correct,
                "pred": best["plate"],
                "gt": gt["plate"],
                "detected": True,
            }
        else:
            per_image[name] = {
                "correct": False,
                "pred": best["plate"] if best else "",
                "gt": gt["plate"],
                "detected": bool(preds),
            }
        if best is not None:
            info = best.get("deskew") or {}
            if info.get("applied"):
                stats["applied"] += 1
                stats["methods"][info.get("method", "?")] += 1
                stats["angles"].append(abs(info.get("angle", 0.0)))
    return stats, per_image


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "-"


def main():
    parser = argparse.ArgumentParser(description="自动拉正前后对比")
    parser.add_argument("--source", default="data/test_images")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="samples/results/deskew_report.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["two_stage"]
    detector = PlateDetector(
        weights=tcfg["detector"]["weights"],
        conf_threshold=tcfg["detector"]["conf_threshold"],
        iou_threshold=tcfg["detector"]["iou_threshold"],
        device=tcfg["detector"]["device"],
    )
    detector.load()
    recognizer = PlateRecognizer(
        weights=tcfg["recognizer"]["weights"],
        device=tcfg["recognizer"]["device"],
    )
    recognizer.load()

    images = list_images(args.source, recursive=False)
    print(f"测试图片：{len(images)} 张")

    base_pipe = TwoStagePipeline(detector, recognizer, None)
    desk_pipe = TwoStagePipeline(detector, recognizer, PlateDeskewer())

    print("跑 baseline ...")
    base_records = evaluate_pipeline(base_pipe, images)
    print("跑 deskew ...")
    desk_records = evaluate_pipeline(desk_pipe, images)

    base_stats, base_img = summarize(base_records)
    desk_stats, desk_img = summarize(desk_records)

    # 分倾斜场景
    scenes = defaultdict(lambda: {"base": 0, "desk": 0, "n": 0})
    for name in base_img:
        try:
            gt = parse_gt(name)
        except ValueError:
            continue
        bucket = bucket_tilt(gt["tilt"])
        scenes[bucket]["n"] += 1
        scenes[bucket]["base"] += int(base_img[name]["correct"])
        scenes[bucket]["desk"] += int(desk_img[name]["correct"])

    fixed = [n for n in base_img if not base_img[n]["correct"] and desk_img[n]["correct"]]
    broken = [n for n in base_img if base_img[n]["correct"] and not desk_img[n]["correct"]]

    mean_base = sum(base_stats["elapsed"]) / len(base_stats["elapsed"]) / 1000
    mean_desk = sum(desk_stats["elapsed"]) / len(desk_stats["elapsed"]) / 1000

    lines = []
    lines.append("# 自动拉正（deskew）前后对比\n")
    lines.append(f"- 测试集：`{args.source}`，有效图片 {base_stats['images']} 张")
    lines.append("- 两条链路共用同一个 YOLO 检测器与 LPRNet 识别器，唯一差别是有无自动拉正")
    lines.append("- 真值仅用于评分；识别链路不使用文件名、不使用任何标注\n")

    lines.append("## 总览\n")
    lines.append("| 指标 | 不拉正 | 自动拉正 | 变化 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| **整牌正确率** | **{pct(base_stats['correct'], base_stats['images'])}** "
        f"（{base_stats['correct']}/{base_stats['images']}） | "
        f"**{pct(desk_stats['correct'], desk_stats['images'])}** "
        f"（{desk_stats['correct']}/{desk_stats['images']}） | "
        f"{desk_stats['correct'] - base_stats['correct']:+d} 张 |"
    )
    lines.append(
        f"| 字符准确率 | {pct(base_stats['char_acc'], base_stats['images'])} | "
        f"{pct(desk_stats['char_acc'], desk_stats['images'])} | "
        f"{(desk_stats['char_acc'] - base_stats['char_acc']) / base_stats['images'] * 100:+.1f} 个百分点 |"
    )
    lines.append(
        f"| 检出率 | {pct(base_stats['detected'], base_stats['images'])} | "
        f"{pct(desk_stats['detected'], desk_stats['images'])} | - |"
    )
    lines.append(
        f"| 平均耗时 | {mean_base:.3f} s/张 | {mean_desk:.3f} s/张 | "
        f"+{(mean_desk - mean_base) * 1000:.0f} ms/张 |"
    )
    lines.append("")
    lines.append("## 分倾斜场景\n")
    lines.append("| 场景 | 图片数 | 不拉正 | 自动拉正 | 变化 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for bucket in TILT_ORDER:
        if bucket not in scenes:
            continue
        s = scenes[bucket]
        lines.append(
            f"| {bucket} | {s['n']} | {pct(s['base'], s['n'])} | {pct(s['desk'], s['n'])} | "
            f"{s['desk'] - s['base']:+d} 张 |"
        )
    lines.append("")

    lines.append("## 拉正触发情况\n")
    lines.append(f"- 触发拉正：{desk_stats['applied']} / {desk_stats['images']} 张"
                 f"（{pct(desk_stats['applied'], desk_stats['images'])}）")
    lines.append("- 触发方式分布：" + "、".join(
        f"`{k}` {v} 张" for k, v in sorted(desk_stats["methods"].items(), key=lambda x: -x[1])
    ))
    if desk_stats["angles"]:
        lines.append(
            f"- 估计倾角：平均 {sum(desk_stats['angles']) / len(desk_stats['angles']):.1f}°，"
            f"最大 {max(desk_stats['angles']):.1f}°"
        )
    lines.append("")

    lines.append("## 逐张变化\n")
    lines.append(f"- 拉正后修好：{len(fixed)} 张")
    lines.append(f"- 拉正后变差：{len(broken)} 张")
    lines.append(f"- 净收益：{len(fixed) - len(broken):+d} 张")
    if fixed:
        lines.append("\n### 修好的案例\n")
        lines.append("| 图片 | 真值 | 不拉正 | 拉正后 |")
        lines.append("| --- | --- | --- | --- |")
        for name in fixed[:25]:
            lines.append(
                f"| {name[:52]} | {base_img[name]['gt']} | {base_img[name]['pred']} | {desk_img[name]['pred']} |"
            )
    if broken:
        lines.append("\n### 变差的案例\n")
        lines.append("| 图片 | 真值 | 不拉正 | 拉正后 |")
        lines.append("| --- | --- | --- | --- |")
        for name in broken[:25]:
            lines.append(
                f"| {name[:52]} | {base_img[name]['gt']} | {base_img[name]['pred']} | {desk_img[name]['pred']} |"
            )

    report = "\n".join(lines) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
