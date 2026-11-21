"""分步式方案：LPRNet 字符识别模块。"""

import cv2
import numpy as np
import torch

from .plate_net import myNet_ocr


class PlateRecognizer:
    """车牌字符识别器，对裁剪出的车牌输出号码字符串。

    网络与权重来自 we0091234 项目（myNet_ocr，LPRNet 式
    CNN + CTC 结构）：输入 168x48，归一化 (x/255 - 0.588) / 0.193，
    字符表以 # 为空白符（下标 0）。权重为带颜色头版本，本项目不用
    颜色输出，加载时忽略其颜色支路参数（见 load 方法）。
    """

    INPUT_WIDTH = 168
    INPUT_HEIGHT = 48

    # 与原项目 plate_rec.py 的 plateName 完全一致，顺序不能动
    CHARS = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
    MEAN = 0.588
    STD = 0.193

    def __init__(self, weights: str, device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self.model = None

    def load(self) -> None:
        """加载识别权重（checkpoint 内含 state_dict 与 cfg）。"""
        checkpoint = torch.load(self.weights, map_location=self.device)
        model = myNet_ocr(
            num_classes=len(self.CHARS),
            export=True,
            cfg=checkpoint["cfg"],
        )
        # strict=False 仅用于忽略权重中的颜色支路参数；字符通路
        # 所需参数若有缺失，直接报错，避免带随机参数静默运行。
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(f"权重缺少字符识别所需参数：{incompatible.missing_keys}")
        model.to(self.device)
        model.eval()
        self.model = model

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """缩放到 168x48，按训练时的均值/方差归一化，返回 CHW 数组。"""
        resized = cv2.resize(plate_image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        img = resized.astype(np.float32)
        img = (img / 255.0 - self.MEAN) / self.STD
        return img.transpose(2, 0, 1)

    @staticmethod
    def _decode(indices) -> str:
        """CTC 风格解码：去掉空白符（0）和连续重复字符。"""
        chars = []
        prev = 0
        for idx in indices:
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(PlateRecognizer.CHARS[idx])
            prev = idx
        return "".join(chars)

    def recognize(self, plate_image: np.ndarray) -> str:
        """识别单张裁剪车牌，返回号码字符串。"""
        if self.model is None:
            raise RuntimeError("识别模型未加载，请先调用 load()。")
        tensor = (
            torch.from_numpy(self.preprocess(plate_image))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            output = self.model(tensor)
        preds = output[0] if isinstance(output, (tuple, list)) else output
        preds = torch.softmax(preds, dim=-1)
        _, indices = preds.max(dim=-1)
        return self._decode(indices.view(-1).cpu().numpy())
