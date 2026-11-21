"""完整识别流程：串联车牌检测与字符识别。"""

from typing import Dict, List

import cv2
import numpy as np

from .detector import PlateDetector
from .recognizer import PlateRecognizer


class PlateRecognitionPipeline:
    """车牌识别完整流程。

    组合 PlateDetector 与 PlateRecognizer，对输入图片
    依次执行车牌定位、裁剪、字符识别，并提供结果可视化。
    """

    def __init__(self, detector: PlateDetector, recognizer: PlateRecognizer) -> None:
        self.detector = detector
        self.recognizer = recognizer

    def run(self, image: np.ndarray) -> List[Dict]:
        """对单张图片执行检测 + 识别。

        参数:
            image: BGR 格式的输入图像。

        返回:
            结果列表，每个元素包含 bbox、confidence、plate。
        """
        detections = self.detector.detect(image)
        results: List[Dict] = []
        for x1, y1, x2, y2, confidence in detections:
            plate_crop = image[y1:y2, x1:x2]
            plate_text = self.recognizer.recognize(plate_crop)
            results.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "plate": plate_text,
                }
            )
        return results

    def draw(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """在图像上绘制车牌框与识别号码。

        参数:
            image: 原始 BGR 图像。
            results: run 方法返回的识别结果。

        返回:
            标注后的图像。
        """
        annotated = image.copy()
        for item in results:
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                item["plate"],
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        return annotated
