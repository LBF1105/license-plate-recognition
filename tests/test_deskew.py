"""自动拉正模块的几何行为测试（不需要模型权重）。"""

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_recognition.two_stage.deskewer import PlateDeskewer, _order_corners


def _tilted_plate(angle_deg: float = 12.0) -> np.ndarray:
    """合成一张黑底 + 倾斜蓝牌 + 若干白色字符的图。"""
    img = np.full((200, 360, 3), 45, np.uint8)
    center = (180.0, 100.0)
    # 车牌自身宽高比约 3.2:1，与真实蓝牌一致
    box = np.float32([[36, 55], [324, 55], [324, 145], [36, 145]])
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    quad = (box @ matrix[:, :2].T) + matrix[:, 2]
    cv2.fillConvexPoly(img, quad.astype(np.int32), (200, 90, 30))
    for i in range(7):
        cx = 70 + i * 36
        cv2.rectangle(img, (cx, 72), (cx + 18, 128), (240, 240, 240), -1)
    return img


def test_candidates_include_original_first():
    image = _tilted_plate()
    candidates = PlateDeskewer().candidates(image)
    assert candidates, "至少应返回原图候选"
    assert candidates[0].applied is False
    assert candidates[0].method == "none"
    assert candidates[0].image is image


def test_candidates_add_deskewed_version_for_tilted_plate():
    image = _tilted_plate(angle_deg=14.0)
    candidates = PlateDeskewer().candidates(image)
    assert len(candidates) >= 2, "倾斜车牌应当产生拉正候选"
    warped = candidates[1]
    assert warped.applied is True
    assert warped.method.startswith("color-")
    assert abs(warped.angle) > 5.0
    # 拉正后的宽高比应接近普通车牌（约 3:1），而不是原图的宽扁比例
    h, w = warped.image.shape[:2]
    assert 1.8 < w / h < 5.0


def test_flat_plate_yields_identity_only():
    """已经水平的车牌不应被重采样，避免无谓插值糊字。"""
    image = _tilted_plate(angle_deg=0.0)
    candidates = PlateDeskewer(min_apply_angle=2.0).candidates(image)
    assert len(candidates) == 1
    assert candidates[0].applied is False


def test_noise_image_falls_back_to_identity():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (80, 240, 3), dtype=np.uint8)
    candidates = PlateDeskewer().candidates(image)
    assert len(candidates) == 1
    assert candidates[0].applied is False


def test_deskew_returns_usable_image():
    image = _tilted_plate(angle_deg=15.0)
    result = PlateDeskewer().deskew(image)
    assert result.image.size > 0
    assert result.image.ndim == 3


def test_order_corners_normalizes_order():
    pts = np.array([[324, 145], [36, 55], [36, 145], [324, 55]], dtype=np.float32)
    ordered = _order_corners(pts)
    assert ordered[0].tolist() == [36, 55]
    assert ordered[1].tolist() == [324, 55]
    assert ordered[2].tolist() == [324, 145]
    assert ordered[3].tolist() == [36, 145]
