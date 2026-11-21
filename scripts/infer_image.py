"""单张图片车牌识别脚本。

用法:
    python scripts/infer_image.py --source data/test_images/test.jpg
"""

import argparse
import sys
from pathlib import Path

# 将 src 目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import yaml

from plate_recognition.detector import PlateDetector
from plate_recognition.pipeline import PlateRecognitionPipeline
from plate_recognition.recognizer import PlateRecognizer


def load_config(config_path: str) -> dict:
    """读取 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(config: dict) -> PlateRecognitionPipeline:
    """根据配置构建检测 + 识别完整流程。"""
    detector = PlateDetector(
        weights=config["detector"]["weights"],
        conf_threshold=config["detector"]["conf_threshold"],
        iou_threshold=config["detector"]["iou_threshold"],
        device=config["detector"]["device"],
    )
    detector.load()

    recognizer = PlateRecognizer(
        weights=config["recognizer"]["weights"],
        device=config["recognizer"]["device"],
    )
    recognizer.load()

    return PlateRecognitionPipeline(detector, recognizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="单张图片车牌识别")
    parser.add_argument("--source", required=True, help="输入图片路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build_pipeline(config)

    image = cv2.imread(args.source)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{args.source}")

    results = pipeline.run(image)
    annotated = pipeline.draw(image, results)

    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{Path(args.source).name}"
    cv2.imwrite(str(output_path), annotated)

    for item in results:
        print(f"识别结果：{item['plate']}（置信度 {item['confidence']:.3f}）")
    print(f"结果图已保存：{output_path}")


if __name__ == "__main__":
    main()
