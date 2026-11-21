"""方案一入口：运行 HyperLPR3 集成式识别。

用法:
    python scripts/run_integrated.py --source data/test_images
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from plate_recognition.integrated.hyperlpr_runner import IntegratedRecognizer


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperLPR3 集成式识别")
    parser.add_argument("--source", required=True, help="单张图片路径或图片文件夹")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["integrated"]
    out_dir = Path(cfg["paths"]["output"]["integrated"])
    out_dir.mkdir(parents=True, exist_ok=True)

    recognizer = IntegratedRecognizer(
        detect_level=icfg["detect_level"], device=icfg["device"])
    recognizer.load()

    # TODO: 解析输入（单图或文件夹），逐张调用 recognizer.recognize，
    # 经 common.visualize 标注后保存到 out_dir，并打印识别号码与耗时
    raise NotImplementedError("集成式识别流程待实现")


if __name__ == "__main__":
    main()
