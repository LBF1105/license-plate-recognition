"""冒烟测试：验证核心模块可导入、基础功能可用。

运行:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_recognition import __version__
from plate_recognition.common.metrics import accuracy
from plate_recognition.two_stage.detector import PlateDetector
from plate_recognition.two_stage.recognizer import PlateRecognizer


def test_version_is_string():
    """版本号应为非空字符串。"""
    assert isinstance(__version__, str) and __version__


def test_preprocess_shape_and_normalization():
    """分步式识别预处理输出应为 3x48x168 的 CHW 数组，并完成标准化。"""
    recognizer = PlateRecognizer(weights="dummy.pth")
    dummy = np.zeros((100, 300, 3), dtype=np.uint8)
    output = recognizer.preprocess(dummy)

    assert output.shape == (3, 48, 168)
    assert output.dtype == np.float32
    # 纯黑图标准化后应恒为 (0 - 0.588) / 0.193
    assert np.allclose(output, (0.0 - 0.588) / 0.193, atol=1e-4)


def test_charset_matches_weights():
    """字符表应为 78 个字符，且空白符 # 位于下标 0。

    字符表与识别权重严格同源，顺序或长度被误改时识别会整体错位，
    这里守住这个契约。
    """
    assert len(PlateRecognizer.CHARS) == 78
    assert PlateRecognizer.CHARS[0] == "#"


def test_ctc_decode_removes_blanks_and_repeats():
    """CTC 解码应去掉空白符与连续重复字符，拼出完整号码。"""

    def idx(ch: str) -> int:
        return PlateRecognizer.CHARS.index(ch)

    # 模拟模型原始输出：含空白符（0）与重复帧
    raw = [
        0,
        idx("湘"),
        idx("湘"),
        0,
        idx("A"),
        idx("1"),
        idx("2"),
        idx("2"),
        idx("3"),
        idx("4"),
        idx("5"),
        0,
    ]
    assert PlateRecognizer._decode(raw) == "湘A12345"


def test_accuracy_calculation():
    """正确率应按整牌完全一致计算。"""
    preds = ["湘A12345", "湘B67890", "错"]
    truth = ["湘A12345", "湘B67890", "湘C00000"]
    assert accuracy(preds, truth) == 2 / 3 * 100


def test_detector_requires_load():
    """检测器未加载权重时调用 detect 应报错，而不是静默失败。"""
    detector = PlateDetector(weights="dummy.pt")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        detector.detect(dummy)


def test_recognizer_requires_load():
    """识别器未加载权重时调用 recognize 应报错，而不是静默失败。"""
    recognizer = PlateRecognizer(weights="dummy.pth")
    dummy = np.zeros((48, 168, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        recognizer.recognize(dummy)
