"""方案一："""



"""方案二：YOLO + LPRNet 分步式识别。"""

from .detector import PlateDetector
from .pipeline import TwoStagePipeline
from .recognizer import PlateRecognizer

__all__ = ["PlateDetector", "PlateRecognizer", "TwoStagePipeline"]

