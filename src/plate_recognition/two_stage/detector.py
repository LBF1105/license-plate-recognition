"""分步式方案：YOLO 车牌检测模块。"""

from typing import List, Tuple

import numpy as np


class PlateDetector:
    """YOLO 车牌检测器，在整图中定位车牌边界框。

    使用 we0091234/yolov8-plate 的 YOLOv8 检测权重，
    类别 0 = single（单层牌），1 = double（双层牌）。
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

        self.model = YOLO(self.weights)

    def detect_detailed(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """检测车牌，返回 (x1, y1, x2, y2, confidence, class_id) 列表。

        class_id 0 = 单层牌，1 = 双层牌，流水线靠它决定是否做双层拼接。
        """
        if self.model is None:
            raise RuntimeError("检测模型未加载，请先调用 load()。")
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        float(box.conf[0]),
                        int(box.cls[0]),
                    )
                )
        return detections

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """检测图像中的车牌。

        返回:
            (x1, y1, x2, y2, confidence) 列表。
        """
        return [d[:5] for d in self.detect_detailed(image)]
