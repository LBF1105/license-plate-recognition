"""批量测试脚本：识别整个文件夹并统计正确率。

用法:
    python scripts/batch_test.py --source data/test_images \
        --ground-truth output/ground_truth.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from infer_image import build_pipeline, load_config

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量车牌识别与正确率统计")
    parser.add_argument("--source", required=True, help="图片文件夹路径")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ground-truth", default=None, help="真值 CSV：文件名,真实号码")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build_pipeline(config)

    image_dir = Path(args.source)
    images = [p for p in sorted(image_dir.iterdir())
              if p.suffix.lower() in IMAGE_SUFFIXES]

    # 读取真值（可选，用于计算正确率）
    truth = {}
    if args.ground_truth:
        with open(args.ground_truth, newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    truth[row[0]] = row[1]

    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    correct = 0
    for img_path in images:
        image = cv2.imread(str(img_path))
        results = pipeline.run(image)
        predicted = results[0]["plate"] if results else ""

        is_correct = ""
        if img_path.name in truth:
            is_correct = predicted == truth[img_path.name]
            correct += int(is_correct)

        records.append((img_path.name, predicted, is_correct))
        cv2.imwrite(str(output_dir / f"result_{img_path.name}"),
                    pipeline.draw(image, results))

    # 保存逐条识别记录
    with open(output_dir / "batch_results.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "识别号码", "是否正确"])
        writer.writerows(records)

    print(f"共测试 {len(images)} 张，结果已保存至 {output_dir}")
    if truth:
        rate = correct / len(images) * 100 if images else 0.0
        print(f"正确 {correct} 张，正确率 {rate:.2f}%")


if __name__ == "__main__":
    main()
