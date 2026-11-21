"""方案一入口：运行 HyperLPR3 集成式识别，支持单图、目录批量与多进程。

用法:
    # 单张
    python scripts/run_integrated.py --source samples/demo2.jpg
    # 批量（分块检测）
    python scripts/run_integrated.py --source data/test_images --tile
    # 大批量：4 进程并行，结果图片已存在则跳过
    python scripts/run_integrated.py --source data/test_images --tile --workers 4 --skip-existing

批量结果会汇总写入 CSV（默认 samples/results/results.csv）。
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import yaml

from plate_recognition.common.image_io import list_images, load_image
from plate_recognition.common.visualize import draw_results
from plate_recognition.integrated.hyperlpr_runner import IntegratedRecognizer

# 进程内复用的识别器（串行模式直接使用，多进程模式由 initializer 初始化）
_WORKER = None


def _init_worker(kwargs: dict) -> None:
    global _WORKER
    _WORKER = IntegratedRecognizer(**kwargs)
    _WORKER.load()


def _process_one(task: tuple) -> dict:
    """处理单张图片：读取 → 识别 → 可选保存结果图。"""
    image_path, out_dir, save = task
    try:
        image = load_image(image_path)
    except Exception as exc:  # 单张坏图不影响整批
        return {"image": image_path, "elapsed_ms": 0.0, "results": [], "error": str(exc)}

    start = time.perf_counter()
    results = _WORKER.recognize(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if save:
        output_path = Path(out_dir) / f"{Path(image_path).stem}_result.jpg"
        cv2.imwrite(str(output_path), draw_results(image, results))

    return {"image": image_path, "elapsed_ms": elapsed_ms, "results": results, "error": None}


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
    """把批量结果写成 CSV：每张图每个车牌一行。"""
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
    parser = argparse.ArgumentParser(description="HyperLPR3 集成式识别（支持批量）")
    parser.add_argument("--source", default="data/test_images", help="单张图片路径或图片文件夹（默认 data/test_images）")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--tile", action="store_true", help="启用分块检测，提升远处小车牌召回")
    parser.add_argument("--recursive", action="store_true", help="递归遍历子目录（数据集按子集分文件夹时用）")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数，大批量时建议 4～8（默认 1）")
    parser.add_argument("--skip-existing", action="store_true", help="结果图已存在则跳过该图（支持断点续跑）")
    parser.add_argument("--no-save", action="store_true", help="不保存结果图，只输出 CSV（大批量提速）")
    parser.add_argument("--csv", default=None, help="结果 CSV 路径，默认 <output>/results.csv")
    parser.add_argument("--output", default=None, help="结果图输出目录，默认取配置")
    args = parser.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["integrated"]
    out_dir = Path(args.output or cfg["paths"]["output"]["integrated"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / "results.csv"

    kwargs = {
        "detect_level": icfg.get("detect_level", "high"),
        "device": icfg.get("device", "cpu"),
        "tile": args.tile or icfg.get("tile", False),
        "tile_grid": tuple(icfg.get("tile_grid", [3, 2])),
        "overlap": icfg.get("overlap", 0.15),
    }

    images = resolve_source(args.source, args.recursive)
    save = not args.no_save

    tasks = []
    skipped = 0
    for image_path in images:
        if args.skip_existing and (out_dir / f"{Path(image_path).stem}_result.jpg").exists():
            skipped += 1
            continue
        tasks.append((image_path, str(out_dir), save))

    if skipped:
        print(f"跳过已处理图片 {skipped} 张，剩余 {len(tasks)} 张")
    if not tasks:
        print("没有需要处理的图片")
        return

    print(f"共 {len(tasks)} 张图片，使用 {args.workers} 个进程，分块={'开' if kwargs['tile'] else '关'}")

    records = []
    started = time.perf_counter()

    if args.workers <= 1:
        _init_worker(kwargs)
        for index, task in enumerate(tasks, 1):
            record = _process_one(task)
            records.append(record)
            _print_progress(index, len(tasks), record)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(kwargs,)
        ) as pool:
            futures = [pool.submit(_process_one, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                _print_progress(index, len(tasks), record)

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


def _print_progress(index: int, total: int, record: dict) -> None:
    name = Path(record["image"]).name
    if record["error"]:
        print(f"[{index}/{total}] {name}  失败: {record['error']}")
        return
    plates = "、".join(r["plate"] for r in record["results"]) or "-"
    print(f"[{index}/{total}] {name}  {len(record['results'])} 个车牌  {record['elapsed_ms']:.0f} ms  {plates}")


if __name__ == "__main__":
    main()
