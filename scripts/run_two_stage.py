"""方案二入口：运行 YOLO + LPRNet 分步式识别，支持单图与目录批量。

用法:
    # 单张
    python scripts/run_two_stage.py --source samples/demo.jpg
    # 批量（默认 data/test_images）
    python scripts/run_two_stage.py --source data/test_images
    # 只出 CSV、不写结果图
    python scripts/run_two_stage.py --source data/test_images --no-save

批量结果会汇总写入 CSV（默认 output/two_stage/results.csv），
列格式与 run_integrated.py 完全一致，可直接用 scripts/evaluate.py 评估。
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import yaml

from plate_recognition.common.image_io import list_images, load_image
from plate_recognition.common.visualize import draw_results
from plate_recognition.two_stage.detector import PlateDetector
from plate_recognition.two_stage.pipeline import TwoStagePipeline
from plate_recognition.two_stage.recognizer import PlateRecognizer


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_source(source: str, recursive: bool):
    """把输入路径解析为图片路径列表（支持单张图片或文件夹）。"""
    path = Path(source)
    if path.is_dir():
        images = list_images(source, recursive=recursive)
        if not images:
            raise SystemExit(f"目录下没有图片：{source}")
        return images
    if path.is_file():
        return [str(path)]
    raise SystemExit(f"找不到路径：{source}")


def write_csv(csv_path: Path, records: list) -> None:
    """把批量结果写成 CSV：每张图每个车牌一行，列与方案一一致。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "plate", "confidence", "x1", "y1", "x2", "y2", "elapsed_ms"])
        for record in records:
            if record["error"]:
                writer.writerow([record["image"], "", "", "", "", "", "", ""])
                continue
            if not record["results"]:
                writer.writerow([record["image"], "", "", "", "", "", "", f"{record['elapsed_ms']:.1f}"])
                continue
            for result in record["results"]:
                x1, y1, x2, y2 = result["bbox"]
                writer.writerow(
                    [
                        record["image"],
                        result["plate"],
                        f"{result['confidence']:.4f}",
                        x1,
                        y1,
                        x2,
                        y2,
                        f"{record['elapsed_ms']:.1f}",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + LPRNet 分步式识别（支持批量）")
    parser.add_argument("--source", default="data/test_images", help="单张图片路径或图片文件夹（默认 data/test_images）")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--recursive", action="store_true", help="递归遍历子目录（数据集按子集分文件夹时用）")
    parser.add_argument("--skip-existing", action="store_true", help="结果图已存在则跳过该图（支持断点续跑）")
    parser.add_argument("--no-save", action="store_true", help="不保存结果图，只输出 CSV")
    parser.add_argument("--csv", default=None, help="结果 CSV 路径，默认 <output>/results.csv")
    parser.add_argument("--output", default=None, help="结果图输出目录，默认取配置")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["two_stage"]
    out_dir = Path(args.output or cfg["paths"]["output"]["two_stage"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / "results.csv"

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

    pipeline = TwoStagePipeline(detector, recognizer)

    images = resolve_source(args.source, args.recursive)
    save = not args.no_save

    tasks = []
    skipped = 0
    for image_path in images:
        if args.skip_existing and (out_dir / f"{Path(image_path).stem}_result.jpg").exists():
            skipped += 1
            continue
        tasks.append(image_path)

    if skipped:
        print(f"跳过已处理图片 {skipped} 张，剩余 {len(tasks)} 张")
    if not tasks:
        print("没有需要处理的图片")
        return

    print(f"共 {len(tasks)} 张图片（分步式为串行处理）")

    records = []
    started = time.perf_counter()
    for index, image_path in enumerate(tasks, 1):
        record = {"image": image_path, "elapsed_ms": 0.0, "results": [], "error": None}
        try:
            image = load_image(image_path)
        except Exception as exc:  # 单张坏图不影响整批
            record["error"] = str(exc)
            records.append(record)
            print(f"[{index}/{len(tasks)}] {Path(image_path).name}  失败: {exc}")
            continue

        start = time.perf_counter()
        record["results"] = pipeline.run(image)
        record["elapsed_ms"] = (time.perf_counter() - start) * 1000
        records.append(record)

        if save:
            output_path = out_dir / f"{Path(image_path).stem}_result.jpg"
            cv2.imwrite(str(output_path), draw_results(image, record["results"]))

        plates = "、".join(r["plate"] for r in record["results"]) or "-"
        print(f"[{index}/{len(tasks)}] {Path(image_path).name}  {len(record['results'])} 个车牌  {record['elapsed_ms']:.0f} ms  {plates}")

    total_seconds = time.perf_counter() - started
    write_csv(csv_path, records)

    failed = [r for r in records if r["error"]]
    total_plates = sum(len(r["results"]) for r in records)
    elapsed_sum = sum(r["elapsed_ms"] for r in records)
    print("\n===== 批量识别汇总 =====")
    print(f"图片总数 : {len(tasks)}（跳过 {skipped}）")
    print(f"识别车牌 : {total_plates} 个")
    print(f"失败图片 : {len(failed)} 张")
    print(f"总耗时   : {total_seconds:.1f} s")
    print(f"平均单张 : {elapsed_sum / len(records) / 1000:.2f} s（推理耗时）")
    print(f"CSV 结果 : {csv_path}")
    if failed:
        print("失败列表：")
        for record in failed:
            print(f"  {record['image']}: {record['error']}")


if __name__ == "__main__":
    main()
