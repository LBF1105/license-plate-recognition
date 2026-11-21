"""分步式方案：YOLO 车牌检测模块。"""

from typing import List, Tuple

import numpy as np


class PlateDetector:
    """YOLO 车牌检测器，在整图中定位车牌边界框。"""

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None

    def load(self) -> None:
        """加载 YOLO 检测权重。"""
        # TODO: from ultralytics import YOLO 并加载权重
        raise NotImplementedError("YOLO 模型加载待实现")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """检测图像中的车牌。

        返回:
            (x1, y1, x2, y2, confidence) 列表。
        """
        if self.model is None:
            raise RuntimeError("检测模型未加载，请先调用 load()。")
        # TODO: 推理并解析边界框坐标
        raise NotImplementedError("YOLO 检测待实现")
