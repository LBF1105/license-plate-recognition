"""CCPD 文件名真值解析脚本。

CCPD 文件名以 "-" 分隔，其中第 5 段为车牌字符索引，
第 1 个索引对应省份，其余索引对应字母与数字。

注意：CCPD 官方字符表是「字母在前、数字在后」（A-Z 共 24 个字母，数字不包含 I、O），
与常见「数字在前」的顺序不同，顺序写反会导致解析出的号码整体错乱。

用法:
    python scripts/parse_ccpd.py --source data/test_images
"""

import argparse
import csv
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

# 省份映射（以 CCPD 官方映射为准）
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "新", "宁", "青",
    "藏", "川", "京", "云", "桂", "琼", "黑", "湘", "甘", "吉",
    "辽", "晋", "鲁", "豫", "鄂", "赣", "苏", "浙", "闽", "粤",
    "贵", "陕",
]

# 字母与数字（CCPD 官方顺序：字母在前、数字在后；车牌不使用 I、O）
CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]


def parse_plate(filename: str) -> str:
    """从 CCPD 文件名解析真实车牌号码。"""
    parts = Path(filename).stem.split("-")
    indices = [int(x) for x in parts[4].split("_")]

    plate = PROVINCES[indices[0]]
    for idx in indices[1:]:
        plate += CHARS[idx]
    return plate


def main() -> None:
    parser = argparse.ArgumentParser(description="CCPD 文件名真值解析")
    parser.add_argument("--source", required=True, help="CCPD 图片文件夹")
    parser.add_argument("--output", default="output/ground_truth.csv")
    args = parser.parse_args()

    image_dir = Path(args.source)
    images = [p for p in sorted(image_dir.iterdir())
              if p.suffix.lower() in IMAGE_SUFFIXES]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "真实号码"])
        for img_path in images:
            writer.writerow([img_path.name, parse_plate(img_path.name)])

    print(f"已解析 {len(images)} 张图片，真值保存至 {output_path}")


if __name__ == "__main__":
    main()
