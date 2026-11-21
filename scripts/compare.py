"""双方案对照入口：对同一组图片分别运行两套方案并对比。

对照维度：整牌识别正确率、单图平均耗时、失败案例。

用法:
    python scripts/compare.py --source data/test_images \
        --ground-truth output/ground_truth.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="双方案对照实验")
    parser.add_argument("--source", required=True, help="同一组测试图片文件夹")
    parser.add_argument("--ground-truth", default=None,
                        help="真值 CSV，用于计算两方案正确率")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    # TODO:
    # 1. 对同一组图片分别运行集成式与分步式方案
    # 2. 用 common.metrics.accuracy 与 Timer 比较正确率、平均耗时
    # 3. 汇总对照表与失败案例，输出到 output/comparison
    raise NotImplementedError("双方案对照待实现")


if __name__ == "__main__":
    main()
