"""图像增强脚本（拓展实验用）：批量对图片做 OpenCV 增强，文件名不变输出到目标目录。

设计原则：增强是独立的前处理步骤，不侵入识别流水线。增强图保持原文件名，
因此 CCPD 文件名中的真值仍然有效，可直接用 run_two_stage.py + evaluate.py
与原图结果做增强前后的对比。

增强方法（可组合，固定按 去噪 → Gamma → CLAHE 的顺序执行）：
    --denoise       轻度去噪（fastNlMeansDenoisingColored，强度取小值避免抹掉笔画）
    --gamma 0.6     Gamma 校正（<1 提亮暗部，>1 压暗）
    --clahe         CLAHE 自适应直方图均衡（在 LAB 的 L 通道上做，只动亮度不动颜色）

CCPD 偏暗子集筛选：
    --brightness-max 100  只处理文件名亮度段（倒数第二段）小于该值的 CCPD 图片；
                          非 CCPD 命名的图片在此模式下跳过。

用法:
    python scripts/enhance_images.py --source data/test_images --output data/enhanced_dark_b ^
        --brightness-max 100 --clahe --gamma 0.6
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plate_recognition.common.image_io import list_images, load_image


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma 校正：用查找表实现，gamma<1 提亮、gamma>1 压暗。"""
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, lut)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE 自适应直方图均衡：转到 LAB 空间只均衡 L（亮度）通道。"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_denoise(image: np.ndarray) -> np.ndarray:
    """轻度去噪：h 取 5，避免把字符笔画当噪声抹掉。"""
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def ccpd_brightness(filename: str):
    """从 CCPD 文件名取亮度值（倒数第二段）；非 CCPD 命名返回 None。"""
    parts = Path(filename).stem.split("-")
    if len(parts) < 7:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="批量图像增强（拓展实验前处理）")
    parser.add_argument("--source", required=True, help="输入图片文件夹")
    parser.add_argument("--output", required=True, help="增强图片输出文件夹")
    parser.add_argument("--clahe", action="store_true", help="启用 CLAHE 对比度增强")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Gamma 校正值，如 0.6（<1 提亮暗部）")
    parser.add_argument("--denoise", action="store_true", help="启用轻度去噪")
    parser.add_argument("--brightness-max", type=int, default=None,
                        help="只处理 CCPD 文件名亮度值小于该值的图片")
    parser.add_argument("--recursive", action="store_true", help="递归处理子文件夹")
    args = parser.parse_args()

    if not (args.clahe or args.gamma is not None or args.denoise):
        parser.error("至少指定一种增强方法：--clahe / --gamma / --denoise")

    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped_filter = 0
    failed = 0
    for img_path in [Path(p) for p in list_images(source, recursive=args.recursive)]:
        if args.brightness_max is not None:
            brightness = ccpd_brightness(img_path.name)
            if brightness is None or brightness >= args.brightness_max:
                skipped_filter += 1
                continue
        try:
            image = load_image(img_path)
            if args.denoise:
                image = apply_denoise(image)
            if args.gamma is not None:
                image = apply_gamma(image, args.gamma)
            if args.clahe:
                image = apply_clahe(image)
        except Exception as exc:  # 单张失败不中断整批
            print(f"跳过（处理失败）：{img_path.name}（{exc}）")
            failed += 1
            continue
        out_path = output_dir / img_path.name
        if not cv2.imwrite(str(out_path), image):
            print(f"跳过（写入失败）：{img_path.name}")
            failed += 1
            continue
        done += 1

    methods = []
    if args.denoise:
        methods.append("去噪")
    if args.gamma is not None:
        methods.append(f"Gamma={args.gamma}")
    if args.clahe:
        methods.append("CLAHE")
    print(f"增强完成（{' + '.join(methods)}）：共输出 {done} 张至 {output_dir}")
    if skipped_filter:
        print(f"按亮度筛选跳过 {skipped_filter} 张")
    if failed:
        print(f"失败 {failed} 张")


if __name__ == "__main__":
    main()
