"""车牌检测模块：基于 YOLO 定位车牌区域。"""

from pathlib import Path
from typing import List, Tuple

import numpy as np


class PlateDetector:
    """YOLO 车牌检测器。

    封装 Ultralytics YOLO 模型，负责在整张车辆图片中
    检测车牌并输出边界框坐标与置信度。
    """

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
        from ultralytics import YOLO

        if not Path(self.weights).exists():
            raise FileNotFoundError(f"未找到检测权重：{self.weights}")
        self.model = YOLO(self.weights)

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """检测图像中的车牌。

        参数:
            image: BGR 格式的输入图像（由 cv2.imread 读取）。

        返回:
            车牌列表，每个元素为 (x1, y1, x2, y2, confidence)。
        """
        if self.model is None:
            raise RuntimeError("检测模型未加载，请先调用 load()。")

        result = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        boxes: List[Tuple[int, int, int, int, float]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        return boxes
