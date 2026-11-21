"""方案二入口：运行 YOLO + LPRNet 分步式识别。

用法:
    python scripts/run_two_stage.py --source data/test_images
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from plate_recognition.two_stage.detector import PlateDetector
from plate_recognition.two_stage.pipeline import TwoStagePipeline
from plate_recognition.two_stage.recognizer import PlateRecognizer


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + LPRNet 分步式识别")
    parser.add_argument("--source", required=True, help="单张图片路径或图片文件夹")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["two_stage"]
    out_dir = Path(cfg["paths"]["output"]["two_stage"])
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # TODO: 解析输入（单图或文件夹），逐张调用 pipeline.run，
    # 经 common.visualize 标注后保存到 out_dir，并打印识别号码与耗时
    raise NotImplementedError("分步式识别流程待实现")


if __name__ == "__main__":
    main()
