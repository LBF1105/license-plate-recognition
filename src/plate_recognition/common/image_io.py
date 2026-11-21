"""图像读取与基础 IO 工具。"""

from pathlib import Path
from typing import List

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def load_image(path: str) -> np.ndarray:
    """读取 BGR 图像，读取失败抛出异常。"""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return image


def list_images(directory: str) -> List[str]:
    """列出目录下所有受支持的图片路径。"""
    folder = Path(directory)
    return [str(p) for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES]
