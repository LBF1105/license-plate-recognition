"""冒烟测试：验证核心模块可导入、基础功能可用。

运行:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_recognition import __version__
from plate_recognition.common.metrics import accuracy
from plate_recognition.two_stage.recognizer import PlateRecognizer


def test_version_is_string():
    """版本号应为非空字符串。"""
    assert isinstance(__version__, str) and __version__


def test_preprocess_shape_and_range():
    """分步式识别预处理输出应为 24x94，且归一化到 [0, 1]。"""
    recognizer = PlateRecognizer(weights="dummy.pth")
    dummy = np.zeros((100, 300, 3), dtype=np.uint8)
    output = recognizer.preprocess(dummy)

    assert output.shape == (24, 94, 3)
    assert 0.0 <= output.min() <= output.max() <= 1.0


def test_accuracy_calculation():
    """正确率应按整牌完全一致计算。"""
    preds = ["湘A12345", "湘B67890", "错"]
    truth = ["湘A12345", "湘B67890", "湘C00000"]
    assert accuracy(preds, truth) == 2 / 3 * 100
