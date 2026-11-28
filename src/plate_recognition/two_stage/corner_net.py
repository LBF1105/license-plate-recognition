"""车牌四角点回归网络（CornerNet）。

作用：给定 YOLO 检测框裁出来的车牌小图，直接回归出车牌的**四个真实顶点**，
再用透视变换把它拉成标准长方形，最后交给 LPRNet 识别。

为什么需要它：YOLOv8 的 detect 头只输出轴对齐矩形 (x, y, w, h)，数学上无法表达
旋转/斜视。本模块用一个很小的卷积网络补上这一维，使倾斜车牌可以被真正拉正。

输入：任意尺寸的 BGR 裁剪图（内部 resize 到 192×64）
输出：4 个角点（左上、右上、右下、左下），坐标为原裁剪图尺度

支持多模型集成：把若干个 .pt 放在同一目录，预测时取角点均值，精度更稳。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

INPUT_W, INPUT_H = 192, 64
OUTPUT_W, OUTPUT_H = 168, 48
# 允许的宽高比范围，超出说明角点回归失败
MIN_RATIO, MAX_RATIO = 1.5, 6.5


def _blk(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class CornerNet(nn.Module):
    """5 层 stride-2 卷积 + 全连接回归 8 个坐标值。"""

    def __init__(self, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            _blk(3, width),
            _blk(width, width * 2),
            _blk(width * 2, width * 3),
            _blk(width * 3, width * 4),
            _blk(width * 4, width * 4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 2 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 8),
        )
        # 输出层初始化成「标准车牌四角」，让训练从合理的起点开始，
        # 否则随机初始化的 8 维输出要花很久才能收敛到车牌形状。
        with torch.no_grad():
            self.fc[-1].bias.copy_(
                torch.tensor([0.06, 0.12, 0.94, 0.05, 0.94, 0.95, 0.06, 0.88])
            )
            self.fc[-1].weight.mul_(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.body(x))


def order_quad(pts: np.ndarray) -> np.ndarray:
    """把任意顺序的四个点排成 左上/右上/右下/左下。"""
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]],
        dtype=np.float32,
    )


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D 叉积（numpy 2.x 的 np.cross 不再接受 2 维向量）。"""
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


@dataclass
class CornerResult:
    """角点预测结果。"""

    corners: np.ndarray          # 4×2，原裁剪图坐标
    valid: bool                  # 是否可用（形状合理）
    reason: str = ""             # 不可用原因


class CornerPredictor:
    """角点回归推理封装，支持加载多个模型做集成。"""

    def __init__(self, weights: Union[str, Path, Sequence[Union[str, Path]]],
                 device: str = "cpu"):
        self.weights = weights
        self.device = device
        self.models: List[CornerNet] = []

    def _resolve(self) -> List[Path]:
        w = self.weights
        if isinstance(w, (list, tuple)):
            return [Path(p) for p in w]
        path = Path(w)
        if path.is_dir():
            files = sorted(path.glob("*.pt"))
            if not files:
                raise FileNotFoundError(f"目录下没有 .pt 模型: {path}")
            return files
        if not path.exists():
            raise FileNotFoundError(f"角点模型不存在: {path}")
        return [path]

    def load(self) -> "CornerPredictor":
        self.models = []
        for path in self._resolve():
            model = CornerNet().to(self.device)
            state = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)
        return self

    def _to_input(self, crop: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return (t - 0.5) / 0.5

    def _predict_raw(self, crop: np.ndarray, h: int, w: int) -> np.ndarray:
        with torch.no_grad():
            inp = self._to_input(crop).unsqueeze(0).to(self.device)
            outs = [m(inp)[0].cpu().numpy().reshape(4, 2) for m in self.models]
        pts = np.mean([order_quad(o) for o in outs], axis=0).astype(np.float32) \
            if len(outs) > 1 else outs[0]
        pts[:, 0] = np.clip(pts[:, 0], -1.5, 2.5) * w
        pts[:, 1] = np.clip(pts[:, 1], -1.5, 2.5) * h
        return order_quad(pts)

    def predict(self, crop: np.ndarray) -> CornerResult:
        """预测四个角点。crop 为 BGR 图。"""
        if not self.models:
            raise RuntimeError("请先调用 load()")
        h, w = crop.shape[:2]
        if h < 6 or w < 12:
            return CornerResult(np.zeros((4, 2), np.float32), False, "too-small")
        pts = self._predict_raw(crop, h, w)

        # 合理性检查：面积、宽高比、高度
        area = 0.5 * abs(_cross2(pts[1] - pts[0], pts[2] - pts[0])) + \
            0.5 * abs(_cross2(pts[2] - pts[0], pts[3] - pts[0]))
        if area < 200:
            return CornerResult(pts, False, "degenerate")
        ew = (np.linalg.norm(pts[0] - pts[1]) + np.linalg.norm(pts[3] - pts[2])) / 2.0
        eh = (np.linalg.norm(pts[0] - pts[3]) + np.linalg.norm(pts[1] - pts[2])) / 2.0
        if eh < 1:
            return CornerResult(pts, False, "zero-height")
        ratio = ew / eh
        if not (MIN_RATIO <= ratio <= MAX_RATIO):
            return CornerResult(pts, False, f"ratio {ratio:.2f}")
        return CornerResult(pts, True)

    @staticmethod
    def warp(crop: np.ndarray, corners: np.ndarray,
             out: Tuple[int, int] = (OUTPUT_W, OUTPUT_H)) -> Optional[np.ndarray]:
        """用四个角点做透视变换，输出拉正后的车牌。"""
        q = order_quad(np.asarray(corners, dtype=np.float32))
        w = max(1.0, (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2.0)
        h = max(1.0, (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2.0)
        if h < 1 or not (MIN_RATIO <= w / h <= MAX_RATIO):
            return None
        scale = out[0] / w
        tw, th = out[0], max(8, int(round(h * scale)))
        if th > 3 * out[1] or th < out[1] // 3:
            return None
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(q, dst)
        warped = cv2.warpPerspective(crop, matrix, (tw, th), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(warped, out, interpolation=cv2.INTER_CUBIC)
