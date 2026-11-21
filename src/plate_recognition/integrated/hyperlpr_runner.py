"""HyperLPR3 集成式识别封装。

HyperLPR3 在框架内部完成车牌检测与字符识别，
输入整张图像即可返回车牌号码与位置，无需手动串联模型。

说明：
- 普通模式：整图直接送入 LicensePlateCatcher。
- 分块模式（tile=True）：把大图切成带重叠的网格逐块识别后合并去重，
  适合"图里车牌多但整体分辨率被压缩、远处小车牌漏检"的场景。
"""

from typing import Dict, List, Tuple

import numpy as np


def _iou(a: List[int], b: List[int]) -> float:
    """计算两个 [x1, y1, x2, y2] 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class IntegratedRecognizer:
    """HyperLPR3 集成式识别器。"""

    def __init__(
        self,
        detect_level: str = "high",
        device: str = "cpu",
        tile: bool = False,
        tile_grid: Tuple[int, int] = (3, 2),
        overlap: float = 0.15,
        iou_threshold: float = 0.5,
    ) -> None:
        self.detect_level = detect_level
        self.device = device
        self.tile = tile
        self.tile_grid = tuple(tile_grid)
        self.overlap = overlap
        self.iou_threshold = iou_threshold
        self.catcher = None

    def load(self) -> None:
        """初始化 HyperLPR3 的 LicensePlateCatcher。"""
        import hyperlpr3 as lpr3

        self._lpr3 = lpr3
        level = (
            lpr3.DETECT_LEVEL_HIGH
            if str(self.detect_level).lower() == "high"
            else lpr3.DETECT_LEVEL_LOW
        )
        self.catcher = lpr3.LicensePlateCatcher(detect_level=level)

    def recognize(self, image: np.ndarray) -> List[Dict]:
        """对整张图像执行端到端识别。

        参数:
            image: BGR 格式输入图像。

        返回:
            结果列表，每个元素包含 plate、confidence、bbox。
        """
        if self.catcher is None:
            raise RuntimeError("识别器未加载，请先调用 load()。")

        if self.tile:
            return self._recognize_tiled(image)

        raw = self.catcher(image)
        return [self._to_result(item) for item in raw]

    @staticmethod
    def _to_result(item) -> Dict:
        """把 HyperLPR3 原始结果 [号码, 置信度, 类型, 框] 转为统一结构。"""
        code, confidence, _plate_type, box = item
        x1, y1, x2, y2 = [int(v) for v in box]
        return {
            "plate": str(code),
            "confidence": float(confidence),
            "bbox": [x1, y1, x2, y2],
        }

    def _recognize_tiled(self, image: np.ndarray) -> List[Dict]:
        """分块识别：网格切片 + 坐标回填 + IoU 去重。"""
        height, width = image.shape[:2]
        nx, ny = self.tile_grid
        pad_x = int(width / nx * self.overlap)
        pad_y = int(height / ny * self.overlap)

        detections: List[Dict] = []
        for gy in range(ny):
            for gx in range(nx):
                x1 = int(gx * width / nx)
                y1 = int(gy * height / ny)
                x2 = min(width, int((gx + 1) * width / nx) + pad_x)
                y2 = min(height, int((gy + 1) * height / ny) + pad_y)

                crop = image[y1:y2, x1:x2]
                for item in self.catcher(crop):
                    result = self._to_result(item)
                    bx1, by1, bx2, by2 = result["bbox"]
                    result["bbox"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                    detections.append(result)

        return self._dedup(detections)

    def _dedup(self, detections: List[Dict]) -> List[Dict]:
        """按框重叠度去重，同一位置保留置信度更高的结果。"""
        kept: List[Dict] = []
        for detect in detections:
            duplicate = False
            for existing in kept:
                if _iou(detect["bbox"], existing["bbox"]) > self.iou_threshold:
                    if detect["confidence"] > existing["confidence"]:
                        existing.update(detect)
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detect)
        return kept
