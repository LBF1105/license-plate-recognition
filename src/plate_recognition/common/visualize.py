"""结果可视化工具：绘制车牌框与识别号码。"""

from typing import List

import cv2
import numpy as np


def draw_results(image: np.ndarray, results: List[dict]) -> np.ndarray:
    """在图像上绘制车牌框与识别号码。

    参数:
        image: 原始 BGR 图像。
        results: 统一结构的识别结果，含 bbox 与 plate。

    返回:
        标注后的图像。
    """
    # TODO: 遍历结果绘制矩形框与文字，两方案共用同一套可视化
    raise NotImplementedError("可视化绘制待实现")
