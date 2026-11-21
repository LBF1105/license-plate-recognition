"""分步式流水线集成测试：需要真实权重与测试图片，缺失时自动跳过。

运行:
    pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
DETECT_WEIGHTS = ROOT / "weights" / "plate_detect.pt"
REC_WEIGHTS = ROOT / "weights" / "lprnet.pth"
IMAGE_DIR = ROOT / "data" / "test_images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

pytestmark = pytest.mark.skipif(
    not (DETECT_WEIGHTS.exists() and REC_WEIGHTS.exists()),
    reason="权重文件未下载，跳过流水线集成测试",
)


def test_pipeline_runs_end_to_end():
    """流水线应能对一张真实图片跑通并返回结构正确的结果。"""
    import cv2
    import numpy as np

    from plate_recognition.two_stage import (
        PlateDetector,
        PlateRecognizer,
        TwoStagePipeline,
    )

    images = [
        p for p in sorted(IMAGE_DIR.glob("*")) if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        pytest.skip("data/test_images 下暂无测试图片")

    detector = PlateDetector(str(DETECT_WEIGHTS), 0.25, 0.45, "cpu")
    detector.load()
    recognizer = PlateRecognizer(str(REC_WEIGHTS), "cpu")
    recognizer.load()
    pipeline = TwoStagePipeline(detector, recognizer)

    img = cv2.imdecode(
        np.fromfile(str(images[0]), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    results = pipeline.run(img)

    assert isinstance(results, list)
    for result in results:
        assert {"bbox", "confidence", "plate"} <= result.keys()
        assert isinstance(result["plate"], str) and result["plate"]
