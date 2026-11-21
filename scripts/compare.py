"""双方案对照报告：读取两方案的识别 CSV，生成并排对照报告。

用法:
    python scripts/compare.py \
        --integrated samples/results/integrated_ccpd.csv \
        --two-stage samples/results/two_stage_ccpd.csv \
        --output samples/results/comparison.md

评估口径与 scripts/evaluate.py 完全一致（直接复用其 load_results / evaluate），
两份 CSV 必须来自同一组图片。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import _mean, _pct, _percentile, evaluate, load_results

NAME_A = "方案一（HyperLPR3 集成式）"
NAME_B = "方案二（YOLOv8 + LPRNet 分步式）"


def _rate(correct: int, total: int) -> float:
    return correct / total * 100 if total else 0.0


def render_comparison(res_a: dict, res_b: dict) -> str:
    sa, sb = res_a["stats"], res_b["stats"]
    lines = []
    lines.append("# 双方案对照报告\n")
    lines.append(f"- 有效图片：方案一 {sa['images']} 张 / 方案二 {sb['images']} 张（应为同一组图片）")
    lines.append(f"- 车牌真值：{sa['gt_plates']} 个")

    lines.append("\n## 总览\n")
    lines.append(f"| 指标 | {NAME_A} | {NAME_B} |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 检出率 | {_pct(sa['detected'], sa['images'])} | {_pct(sb['detected'], sb['images'])} |")
    lines.append(
        f"| 检测 TP / FP / FN | {sa['tp']} / {sa['fp']} / {sa['fn']} | {sb['tp']} / {sb['fp']} / {sb['fn']} |"
    )
    lines.append(
        f"| 检测精确率 / 召回率 | {_pct(sa['tp'], sa['tp'] + sa['fp'])} / {_pct(sa['tp'], sa['tp'] + sa['fn'])} "
        f"| {_pct(sb['tp'], sb['tp'] + sb['fp'])} / {_pct(sb['tp'], sb['tp'] + sb['fn'])} |"
    )
    lines.append(
        f"| **整牌正确率** | **{_pct(sa['plate_correct'], sa['images'])}**（{sa['plate_correct']}/{sa['images']}） "
        f"| **{_pct(sb['plate_correct'], sb['images'])}**（{sb['plate_correct']}/{sb['images']}） |"
    )
    lines.append(f"| 字符准确率 | {_pct(sa['char_acc_sum'], sa['images'])} | {_pct(sb['char_acc_sum'], sb['images'])} |")
    lines.append(
        f"| 平均耗时 | {_mean(sa['elapsed']) / 1000:.2f} s/张（P50 {_percentile(sa['elapsed'], 0.5) / 1000:.2f} s，"
        f"P95 {_percentile(sa['elapsed'], 0.95) / 1000:.2f} s） "
        f"| {_mean(sb['elapsed']) / 1000:.2f} s/张（P50 {_percentile(sb['elapsed'], 0.5) / 1000:.2f} s，"
        f"P95 {_percentile(sb['elapsed'], 0.95) / 1000:.2f} s） |"
    )

    lines.append("\n## 分场景整牌正确率\n")
    lines.append("| 维度 | 场景 | 图片数 | 方案一 | 方案二 | 差（百分点） |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    scene_keys = sorted(set(res_a["scenes"]) | set(res_b["scenes"]))
    for key in scene_keys:
        scene_a = res_a["scenes"].get(key)
        scene_b = res_b["scenes"].get(key)
        if scene_a is None or scene_b is None:
            continue
        diff = _rate(scene_a["correct"], scene_a["images"]) - _rate(scene_b["correct"], scene_b["images"])
        lines.append(
            f"| {key[0]} | {key[1]} | {scene_a['images']} | "
            f"{_pct(scene_a['correct'], scene_a['images'])} | {_pct(scene_b['correct'], scene_b['images'])} | {diff:+.1f} |"
        )

    fail_a = {name: (gt, pred) for name, gt, pred, _ in res_a["failures"]}
    fail_b = {name: (gt, pred) for name, gt, pred, _ in res_b["failures"]}
    both = sorted(fail_a.keys() & fail_b.keys())
    only_a = sorted(fail_a.keys() - fail_b.keys())
    only_b = sorted(fail_b.keys() - fail_a.keys())

    lines.append("\n## 失败交集\n")
    lines.append(f"- 两方案共同失败：{len(both)} 张")
    lines.append(f"- 仅方案一失败：{len(only_a)} 张")
    lines.append(f"- 仅方案二失败：{len(only_b)} 张")

    if both:
        lines.append("\n### 共同失败（前 20 条）\n")
        lines.append("| 图片 | 真值 | 方案一预测 | 方案二预测 |")
        lines.append("| --- | --- | --- | --- |")
        for name in both[:20]:
            gt = fail_a[name][0]
            lines.append(f"| {name} | {gt} | {fail_a[name][1]} | {fail_b[name][1]} |")
    if only_a:
        lines.append("\n### 仅方案一失败（前 20 条）\n")
        lines.append("| 图片 | 真值 | 方案一预测 |")
        lines.append("| --- | --- | --- |")
        for name in only_a[:20]:
            lines.append(f"| {name} | {fail_a[name][0]} | {fail_a[name][1]} |")
    if only_b:
        lines.append("\n### 仅方案二失败（前 20 条）\n")
        lines.append("| 图片 | 真值 | 方案二预测 |")
        lines.append("| --- | --- | --- |")
        for name in only_b[:20]:
            lines.append(f"| {name} | {fail_b[name][0]} | {fail_b[name][1]} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="双方案对照报告（基于两份识别 CSV）")
    parser.add_argument("--integrated", required=True, help="方案一（集成式）的结果 CSV")
    parser.add_argument("--two-stage", required=True, help="方案二（分步式）的结果 CSV")
    parser.add_argument("--output", default=None, help="对照报告（markdown）输出路径")
    parser.add_argument("--iou", type=float, default=0.5, help="检测匹配 IoU 阈值（默认 0.5）")
    args = parser.parse_args()

    res_a = evaluate(load_results(Path(args.integrated)), iou_threshold=args.iou)
    res_b = evaluate(load_results(Path(args.two_stage)), iou_threshold=args.iou)
    report = render_comparison(res_a, res_b)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"报告已保存：{output_path}")
    print(report)


if __name__ == "__main__":
    main()
