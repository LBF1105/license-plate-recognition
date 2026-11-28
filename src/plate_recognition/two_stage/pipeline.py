"""分步式方案：串联检测与识别的完整流程。"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .corner_net import CornerPredictor
from .deskewer import PlateDeskewer
from .detector import PlateDetector
from .double_plate_split_merge import get_split_merge
from .recognizer import PlateRecognizer


class TwoStagePipeline:
    """YOLO 检测 + 角点拉正 + LPRNet 识别完整流程。

    车牌被斜着拍时，YOLO 给出的轴对齐框无法表达旋转，直接把框裁出来的图
    会带回大量背景、字符被压扁。这里的做法是：

    1. YOLO 只负责「车牌大致在哪」；
    2. 角点回归网络（CornerPredictor）在裁剪图里回归出车牌的**四个真实顶点**；
    3. 按四个顶点做透视变换，把平行四边形拉成长方形；
    4. 原图和拉正图各识别一次，用识别器自己的 CTC 置信度选更可信的那个。

    第 4 步很关键：LPRNet 对轻中度倾斜本来就有一定鲁棒性，单路强行替换
    会掉点；多候选择优则只赚不赔，倾斜样本的收益最明显。

    角点网络不可用时（没有权重文件，或回归结果形状不合理），会退回
    纯几何的 PlateDeskewer（颜色连通域 + 霍夫直线），再不行就只用原图。
    """

    # 裁剪时向外扩的边距比例，避免框太紧切掉字符
    CROP_MARGIN = 0.05

    def __init__(
        self,
        detector: PlateDetector,
        recognizer: PlateRecognizer,
        deskewer: Optional[PlateDeskewer] = None,
        corner_predictor: Optional[CornerPredictor] = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.deskewer = deskewer
        self.corner_predictor = corner_predictor

    def _corner_candidate(self, crop: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """用角点网络把车牌拉正；失败返回 None。"""
        if self.corner_predictor is None:
            return None
        result = self.corner_predictor.predict(crop)
        if not result.valid:
            return None
        warped = CornerPredictor.warp(crop, result.corners)
        if warped is None:
            return None
        return warped, {
            "applied": True,
            "method": "corner-quad",
            "angle": 0.0,
            "quality": 1.0,
            "corners": result.corners,
        }

    def _candidates(self, crop: np.ndarray, cls_id: int) -> List[Tuple[np.ndarray, Dict]]:
        """裁剪图 → 若干待识别图像及其校正信息。

        双层牌（cls_id == 1）要在每个候选上先做上下两行拼行，
        保证比较的是同一条流水线的产物。
        """
        prepared: List[Tuple[np.ndarray, Dict]] = []

        # 原图永远作为候选保底
        prepared.append((crop, {"applied": False, "method": "none", "angle": 0.0,
                                "quality": 0.0, "corners": None}))

        corner = self._corner_candidate(crop)
        if corner is not None:
            prepared.append(corner)
        elif self.deskewer is not None:
            # 角点网络没给出合理结果时，退回纯几何估计
            for r in self.deskewer.candidates(crop)[1:]:
                prepared.append((r.image, {"applied": r.applied, "method": r.method,
                                           "angle": r.angle, "quality": r.quality,
                                           "corners": r.corners}))

        if cls_id == 1:
            prepared = [(get_split_merge(img), info) for img, info in prepared]
        return prepared

    def run(self, image: np.ndarray) -> List[Dict]:
        """执行检测 → 裁剪 → 角点拉正 → 置信度择优 → 输出号码。

        返回:
            结果列表，每个元素含 bbox、confidence、plate、plate_type，
            以及 deskew（最终采纳的校正方式/质量分）供分析与可视化。
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

            best_plate = ""
            best_score = -1.0
            best_info: Dict = {"applied": False, "method": "none", "angle": 0.0, "quality": 0.0}
            best_corners = None
            for plate_image, info in self._candidates(crop, cls_id):
                plate, score = self.recognizer.recognize_with_score(plate_image)
                if score > best_score:
                    best_plate, best_score = plate, score
                    best_corners = info.get("corners")
                    best_info = {
                        "applied": info.get("applied", False),
                        "method": info.get("method", "none"),
                        "angle": info.get("angle", 0.0),
                        "quality": info.get("quality", 0.0),
                    }

            results.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "plate": best_plate,
                    "plate_type": "double" if cls_id == 1 else "single",
                    "deskew": best_info,
                    "corners": None
                    if best_corners is None
                    else np.asarray(best_corners, dtype=np.float32)
                    + np.array([crop_x1, crop_y1], dtype=np.float32),
                    "recognize_score": best_score,
                }
            )
        return results
