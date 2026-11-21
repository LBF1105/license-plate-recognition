"""字符识别模块：基于 LPRNet 识别车牌字符。"""

from pathlib import Path

import cv2
import numpy as np


class PlateRecognizer:
    """LPRNet 车牌字符识别器。

    对裁剪出的车牌图像进行预处理后送入 LPRNet，
    经 CTC 解码输出完整车牌字符串，无需字符分割。
    """

    # LPRNet 标准输入尺寸
    INPUT_WIDTH = 94
    INPUT_HEIGHT = 24

    def __init__(self, weights: str, device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self.model = None

    def load(self) -> None:
        """加载 LPRNet 权重。"""
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"未找到识别权重：{self.weights}")
        # TODO: 导入 LPRNet 模型定义，加载 .pth 权重并切换到 eval 模式
        raise NotImplementedError("LPRNet 模型加载待实现")

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """车牌图像预处理：缩放到模型输入尺寸并归一化。

        参数:
            plate_image: 裁剪出的车牌 BGR 图像。

        返回:
            预处理后的浮点图像，像素值归一化到 [0, 1]。
        """
        resized = cv2.resize(plate_image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        return resized.astype(np.float32) / 255.0

    def recognize(self, plate_image: np.ndarray) -> str:
        """识别单张车牌图像。

        参数:
            plate_image: 裁剪出的车牌 BGR 图像。

        返回:
            车牌号码字符串。
        """
        if self.model is None:
            raise RuntimeError("识别模型未加载，请先调用 load()。")
        # TODO: 前向推理并执行 CTC 解码，返回车牌字符串
        raise NotImplementedError("LPRNet 字符识别待实现")
