"""结果可视化工具：绘制车牌框与识别号码。"""

from typing import Dict, List

import cv2
import numpy as np

# OpenCV 的 putText 不支持中文字符，这里把省份简称转成拼音，
# 保证在不同平台（Windows / Linux / macOS）上都能正常显示。
_PROVINCE_PINYIN = {
    "京": "Jing", "津": "Jin", "冀": "Ji", "晋": "Jin", "蒙": "Meng",
    "辽": "Liao", "吉": "Ji", "黑": "Hei", "沪": "Hu", "苏": "Su",
    "浙": "Zhe", "皖": "Wan", "闽": "Min", "赣": "Gan", "鲁": "Lu",
    "豫": "Yu", "鄂": "E", "湘": "Xiang", "粤": "Yue", "桂": "Gui",
    "琼": "Qiong", "川": "Chuan", "贵": "Gui", "云": "Yun", "渝": "Yu",
    "藏": "Zang", "陕": "Shan", "甘": "Gan", "青": "Qing", "宁": "Ning",
    "新": "Xin",
}


def plate_to_ascii(plate: str) -> str:
    """把中文车牌号转成可被 OpenCV 绘制的拼音形式，如 冀A9F3J6 -> JiA9F3J6。"""
    return "".join(_PROVINCE_PINYIN.get(char, char) for char in plate)


def draw_results(image: np.ndarray, results: List[Dict]) -> np.ndarray:
    """在图像上绘制车牌框与识别号码。

    参数:
        image: 原始 BGR 图像。
        results: 统一结构的识别结果，含 bbox 与 plate。

    返回:
        标注后的图像。
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, width // 300)
    font_scale = max(0.6, width / 2000)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        label = f"{plate_to_ascii(result['plate'])} {result['confidence']:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_y = max(y1 - 8, text_h + 8)
        cv2.rectangle(
            canvas,
            (x1, text_y - text_h - 6),
            (x1 + text_w + 8, text_y + baseline + 4),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            canvas, label, (x1 + 4, text_y), font, font_scale, (0, 0, 0), thickness
        )

    return canvas
