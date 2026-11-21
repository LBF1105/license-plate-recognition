"""方案一入口：运行 HyperLPR3 集成式识别。

用法:
    python scripts/run_integrated.py --source data/test_images
    python scripts/run_integrated.py --source data/test_images/demo2.jpg --tile
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import yaml

from plate_recognition.common.image_io import list_images, load_image
from plate_recognition.common.visualize import draw_results
from plate_recognition.integrated.hyperlpr_runner import IntegratedRecognizer


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_source(source: str):
    """把输入路径解析为图片路径列表（支持单张图片或文件夹）。"""
    path = Path(source)
    if path.is_dir():
        images = list_images(source)
        if not images:
            raise SystemExit(f"目录下没有图片：{source}")
        return images
    if path.is_file():
        return [str(path)]
    raise SystemExit(f"找不到路径：{source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperLPR3 集成式识别")
    parser.add_argument("--source", required=True, help="单张图片路径或图片文件夹")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--tile",
        action="store_true",
        help="启用分块检测（大图中远处小车牌漏检时使用）",
    )
    parser.add_argument("--output", default=None, help="结果图输出目录，默认取配置")
    args = parser.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["integrated"]
    out_dir = Path(args.output or cfg["paths"]["output"]["integrated"])
    out_dir.mkdir(parents=True, exist_ok=True)

    recognizer = IntegratedRecognizer(
        detect_level=icfg.get("detect_level", "high"),
        device=icfg.get("device", "cpu"),
        tile=args.tile or icfg.get("tile", False),
        tile_grid=tuple(icfg.get("tile_grid", [3, 2])),
        overlap=icfg.get("overlap", 0.15),
    )
    recognizer.load()

    images = resolve_source(args.source)
    total_ms = 0.0

    for image_path in images:
        image = load_image(image_path)

        start = time.perf_counter()
        results = recognizer.recognize(image)
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_ms += elapsed_ms

        output_path = out_dir / f"{Path(image_path).stem}_result.jpg"
        cv2.imwrite(str(output_path), draw_results(image, results))

        print(f"[{Path(image_path).name}] 识别到 {len(results)} 个车牌，耗时 {elapsed_ms:.1f} ms")
        for result in results:
            print(f"    {result['plate']}  置信度 {result['confidence']:.4f}  位置 {result['bbox']}")
        print(f"    结果图: {output_path}")

    if images:
        print(f"\n共 {len(images)} 张图片，平均耗时 {total_ms / len(images):.1f} ms/张")


if __name__ == "__main__":
    main()
