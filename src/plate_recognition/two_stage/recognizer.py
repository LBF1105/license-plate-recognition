"""分步式方案：LPRNet 字符识别模块。"""

import cv2
import numpy as np


class PlateRecognizer:
    """LPRNet 字符识别器，对裁剪出的车牌输出号码字符串。"""

    INPUT_WIDTH = 94
    INPUT_HEIGHT = 24

    def __init__(self, weights: str, device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self.model = None

    def load(self) -> None:
        """加载 LPRNet 权重。"""
        # TODO: 导入 LPRNet 模型定义并加载 .pth
        raise NotImplementedError("LPRNet 模型加载待实现")

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """缩放到 94x24 并归一化到 [0, 1]。"""
        resized = cv2.resize(plate_image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        return resized.astype(np.float32) / 255.0

    def recognize(self, plate_image: np.ndarray) -> str:
        """识别单张裁剪车牌，返回号码字符串。"""
        if self.model is None:
            raise RuntimeError("识别模型未加载，请先调用 load()。")
        # TODO: 前向推理并执行 CTC 解码
        raise NotImplementedError("LPRNet 识别待实现")
