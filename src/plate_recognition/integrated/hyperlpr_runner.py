"""HyperLPR3 集成式识别封装。

HyperLPR3 在框架内部完成车牌检测与字符识别，
输入整张图像即可返回车牌号码与位置，无需手动串联模型。
"""

from typing import Dict, List

import numpy as np


class IntegratedRecognizer:
    """HyperLPR3 集成式识别器。"""

    def __init__(self, detect_level: str = "high", device: str = "cpu") -> None:
        self.detect_level = detect_level
        self.device = device
        self.catcher = None

    def load(self) -> None:
        """初始化 HyperLPR3 的 LicensePlateCatcher。"""
        # TODO: import hyperlpr3 并构造 LicensePlateCatcher
        raise NotImplementedError("HyperLPR3 初始化待实现")

    def recognize(self, image: np.ndarray) -> List[Dict]:
        """对整张图像执行端到端识别。

        参数:
            image: BGR 格式输入图像。

        返回:
            结果列表，每个元素包含 plate、confidence、bbox。
        """
        if self.catcher is None:
            raise RuntimeError("识别器未加载，请先调用 load()。")
        # TODO: 调用 HyperLPR3 推理并整理为统一输出结构
        raise NotImplementedError("HyperLPR3 识别待实现")
