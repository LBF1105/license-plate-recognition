"""分步式方案：串联检测与识别的完整流程。"""

from typing import Dict, List

import numpy as np

from .detector import PlateDetector
from .recognizer import PlateRecognizer


class TwoStagePipeline:
    """YOLO 检测 + LPRNet 识别完整流程。"""

    def __init__(self, detector: PlateDetector, recognizer: PlateRecognizer) -> None:
        self.detector = detector
        self.recognizer = recognizer

    def run(self, image: np.ndarray) -> List[Dict]:
        """执行检测 → 裁剪 → 识别。

        返回:
            结果列表，每个元素含 bbox、confidence、plate。
        """
        # TODO: 调用 detector.detect，逐框裁剪后交给 recognizer.recognize
        raise NotImplementedError("分步式流程待实现")
