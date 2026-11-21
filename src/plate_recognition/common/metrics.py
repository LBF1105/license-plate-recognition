"""评估指标：正确率统计与耗时测量。"""

from typing import List


def accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """计算整牌识别正确率：字符串完全一致才计为正确。

    参数:
        predictions: 模型识别号码列表。
        ground_truth: 真实车牌号码列表。

    返回:
        正确率（百分比）。
    """
    if not ground_truth:
        return 0.0
    correct = sum(int(p == g) for p, g in zip(predictions, ground_truth))
    return correct / len(ground_truth) * 100.0


class Timer:
    """上下文管理器，用于测量单图识别耗时（秒）。"""

    def __enter__(self) -> "Timer":
        self.start = __import__("time").perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = __import__("time").perf_counter() - self.start
