"""任务二汇总脚本：读取四人亲测真值与两方案结果，生成每人一行的正确率统计表。

输入（每个成员目录下）：
    labels.csv               手工真值，列 filename,true_plate,scene
    result_integrated.csv    方案一结果（run_integrated.py 生成）
    result_two_stage.csv     方案二结果（run_two_stage.py 生成）

判定：整牌字符全部一致才算正确；同一文件有多个预测时，任一预测命中真值即算对。

用法:
    python scripts/aggregate_field_test.py --field-root data/field_test \
        --output samples/results/field_test_summary.md
"""

import argparse
import csv
from pathlib import Path


def load_labels(path: Path) -> dict:
    """读取 labels.csv，返回 {文件名: (真实号码, 场景)}。"""
    labels = {}
    if not path.exists():
        return labels
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            filename = (row.get("filename") or "").strip()
            true_plate = (row.get("true_plate") or "").strip()
            scene = (row.get("scene") or "").strip()
            if filename and true_plate:
                labels[filename] = (true_plate, scene)
    return labels


def load_predictions(path: Path) -> dict:
    """读取结果 CSV，按文件名 basename 分组，返回 {文件名: [预测号码,...]}。"""
    preds = {}
    if not path.exists():
        return preds
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = Path((row.get("image") or "").strip()).name
            plate = (row.get("plate") or "").strip()
            if name and plate:
                preds.setdefault(name, []).append(plate)
    return preds


def is_correct(true_plate: str, preds: list) -> bool:
    return any(p == true_plate for p in preds)


def pct(correct: int, total: int) -> str:
    return f"{correct / total * 100:.1f}%" if total else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总四人亲测正确率")
    parser.add_argument("--field-root", default="data/field_test", help="亲测记录根目录")
    parser.add_argument("--output", default="samples/results/field_test_summary.md",
                        help="汇总表 markdown 输出路径")
    args = parser.parse_args()

    root = Path(args.field_root)
    member_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)

    rows = []
    warnings = []
    totals = {"n": 0, "int": 0, "two": 0}

    for member_dir in member_dirs:
        labels = load_labels(member_dir / "labels.csv")
        int_preds = load_predictions(member_dir / "result_integrated.csv")
        two_preds = load_predictions(member_dir / "result_two_stage.csv")

        if not labels:
            warnings.append(f"- {member_dir.name}: 缺少 labels.csv 或内容为空，已跳过")
            continue

        n = len(labels)
        int_correct = sum(is_correct(tp, int_preds.get(name, []))
                          for name, (tp, _) in labels.items())
        two_correct = sum(is_correct(tp, two_preds.get(name, []))
                          for name, (tp, _) in labels.items())

        if not int_preds:
            warnings.append(f"- {member_dir.name}: 缺少 result_integrated.csv")
        if not two_preds:
            warnings.append(f"- {member_dir.name}: 缺少 result_two_stage.csv")

        rows.append((member_dir.name, n, int_correct, two_correct))
        totals["n"] += n
        totals["int"] += int_correct
        totals["two"] += two_correct

    lines = ["# 四人亲测记录汇总\n"]
    lines.append("| 成员 | 测试数 | 方案一正确数 | 方案一正确率 | 方案二正确数 | 方案二正确率 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, n, int_correct, two_correct in rows:
        lines.append(
            f"| {name} | {n} | {int_correct} | {pct(int_correct, n)} "
            f"| {two_correct} | {pct(two_correct, n)} |"
        )
    if totals["n"]:
        lines.append(
            f"| **合计** | **{totals['n']}** | **{totals['int']}** "
            f"| **{pct(totals['int'], totals['n'])}** | **{totals['two']}** "
            f"| **{pct(totals['two'], totals['n'])}** |"
        )

    if warnings:
        lines.append("\n## 数据缺失提示\n")
        lines.extend(warnings)

    report = "\n".join(lines) + "\n"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"汇总表已保存：{output_path}")


if __name__ == "__main__":
    main()
