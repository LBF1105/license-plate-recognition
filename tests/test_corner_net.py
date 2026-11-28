"""角点回归网络的单元测试（不依赖训练好的权重）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import pytest
import torch

from plate_recognition.two_stage.corner_net import (INPUT_H, INPUT_W, CornerNet,
                                                    CornerPredictor, order_quad)


def test_corner_net_forward_shape():
    model = CornerNet()
    out = model(torch.zeros(2, 3, INPUT_H, INPUT_W))
    assert out.shape == (2, 8)


def test_output_bias_prior_is_plate_like():
    """输出层初始偏置应当是「左上/右上/右下/左下」四个角，而不是全零。"""
    model = CornerNet()
    bias = model.fc[-1].bias.detach().numpy().reshape(4, 2)
    assert bias[0][0] < 0.3 and bias[0][1] < 0.3      # 左上
    assert bias[1][0] > 0.7 and bias[1][1] < 0.3      # 右上
    assert bias[2][0] > 0.7 and bias[2][1] > 0.7      # 右下
    assert bias[3][0] < 0.3 and bias[3][1] > 0.7      # 左下


def test_order_quad_orders_corners():
    pts = np.array([[100, 90], [10, 20], [110, 10], [20, 80]], dtype=np.float32)
    q = order_quad(pts)
    assert tuple(q[0]) == (10, 20)    # 左上
    assert tuple(q[1]) == (110, 10)   # 右上
    assert tuple(q[2]) == (100, 90)   # 右下
    assert tuple(q[3]) == (20, 80)    # 左下


def test_warp_outputs_fixed_size():
    img = np.zeros((60, 180, 3), np.uint8)
    quad = np.array([[10, 12], [170, 8], [172, 50], [12, 54]], np.float32)
    out = CornerPredictor.warp(img, quad)
    assert out is not None
    assert out.shape[:2] == (48, 168)


def test_warp_of_rotated_quad_is_level():
    """把一块倾斜矩形拉正后，四个角的亮度分布应当变成水平的。"""
    img = np.zeros((120, 240, 3), np.uint8)
    quad = np.array([[40, 30], [210, 10], [220, 60], [50, 80]], np.float32)
    cv2.fillPoly(img, [quad.astype(np.int32)], (255, 255, 255))
    out = CornerPredictor.warp(img, quad)
    assert out is not None
    # 拉正后车牌应该占满画面，白像素比例 > 0.8
    white = (out.mean(axis=2) > 128).mean()
    assert white > 0.8


def test_warp_rejects_bad_ratio():
    img = np.zeros((60, 180, 3), np.uint8)
    quad = np.array([[10, 10], [20, 10], [20, 50], [10, 50]], np.float32)  # 比例 0.25
    assert CornerPredictor.warp(img, quad) is None


def test_predictor_requires_load():
    predictor = CornerPredictor("nope.pt")
    with pytest.raises(RuntimeError):
        predictor.predict(np.zeros((60, 180, 3), np.uint8))


def test_predictor_missing_weights():
    predictor = CornerPredictor("/tmp/__definitely_missing_corner__.pt")
    with pytest.raises(FileNotFoundError):
        predictor.load()


def test_predictor_rejects_tiny_crop():
    predictor = CornerPredictor("nope.pt")
    predictor.models = [CornerNet()]   # 绕过 load，测尺寸保护
    result = predictor.predict(np.zeros((4, 8, 3), np.uint8))
    assert result.valid is False
    assert result.reason == "too-small"
