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


def list_images(directory: str, recursive: bool = False) -> List[str]:
    """列出目录下的受支持图片路径。

    参数:
        directory: 图片目录。
        recursive: 为 True 时递归遍历子目录（按子集分文件夹的数据集常用）。
    """
    folder = Path(directory)
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return [str(p) for p in sorted(paths) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
