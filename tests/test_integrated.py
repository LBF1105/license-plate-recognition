"""集成式模块的轻量单元测试（不加载模型权重，可离线运行）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_recognition.common.visualize import plate_to_ascii
from plate_recognition.integrated.hyperlpr_runner import IntegratedRecognizer, _iou


def test_plate_to_ascii():
    assert plate_to_ascii("冀A9F3J6") == "JiA9F3J6"
    assert plate_to_ascii("苏E67A12") == "SuE67A12"
    assert plate_to_ascii("京A12345") == "JingA12345"


def test_iou():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert 0 < _iou([0, 0, 10, 10], [5, 5, 15, 15]) < 1


def test_dedup_keeps_highest_confidence():
    recognizer = IntegratedRecognizer()
    detections = [
        {"plate": "A", "confidence": 0.8, "bbox": [0, 0, 10, 10]},
        {"plate": "B", "confidence": 0.9, "bbox": [1, 1, 11, 11]},
        {"plate": "C", "confidence": 0.7, "bbox": [100, 100, 120, 120]},
    ]
    kept = recognizer._dedup(detections)
    assert len(kept) == 2
    assert {item["plate"] for item in kept} == {"B", "C"}
