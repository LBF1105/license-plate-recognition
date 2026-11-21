"""分步式方案：串联检测与识别的完整流程。"""

from typing import Dict, List

import numpy as np

from .detector import PlateDetector
from .double_plate_split_merge import get_split_merge
from .recognizer import PlateRecognizer


class TwoStagePipeline:
    """YOLO 检测 + LPRNet 识别完整流程。"""

    # 裁剪时向外扩的边距比例，避免框太紧切掉字符
    CROP_MARGIN = 0.05

    def __init__(self, detector: PlateDetector, recognizer: PlateRecognizer) -> None:
        self.detector = detector
        self.recognizer = recognizer

    def run(self, image: np.ndarray) -> List[Dict]:
        """执行检测 → 裁剪 → 识别。

        返回:
            结果列表，每个元素含 bbox、confidence、plate，
            另带 plate_type（single / double）供日志与分析使用。
        """
        img_h, img_w = image.shape[:2]
        results = []
        for x1, y1, x2, y2, conf, cls_id in self.detector.detect_detailed(image):
            margin_x = int((x2 - x1) * self.CROP_MARGIN)
            margin_y = int((y2 - y1) * self.CROP_MARGIN)
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(img_w, x2 + margin_x)
            crop_y2 = min(img_h, y2 + margin_y)
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue
            if cls_id == 1:
                # 双层牌：先把上下两行切开拼成一行再识别
                crop = get_split_merge(crop)
            plate = self.recognizer.recognize(crop)
            results.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "plate": plate,
                    "plate_type": "double" if cls_id == 1 else "single",
                }
            )
        return results
