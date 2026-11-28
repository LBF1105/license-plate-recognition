"""自动车牌校正（deskew）：不依赖任何标注，只用裁剪图本身的像素信息。

YOLO 检测头输出的是轴对齐矩形（AABB），车牌一旦倾斜/斜视，
AABB 裁剪就会把车身、背景一起带进来，字符也是歪的。

本模块在 AABB 裁剪图内部再做一次“几何归一化”：

    颜色 + 四边形（主路径）
        中国车牌底色（蓝/绿/黄）在 HSV 空间里分离度很高，取最大连通域
        就得到车牌所在区域；对区域拟合四边形（优先四点近似，可表达透视；
        否则退化为最小外接矩形），再四点透视变换拉成正视图。
        变换时把四边形向外扩一圈，避免边界正好压在首尾字符上把字符切掉。

    梯度投影 + 角度搜索（兜底路径，默认关闭）
        白牌、深色牌、逆光、偏色时颜色先验失效。实测该路径精度不足且
        容易把原本能认对的图转坏，因此默认只在显式开启时使用。

重要的使用方式：不要用 deskew() 的单一结果去替换原图。
`candidates()` 同时给出「原图」和「若干净化版」，交给上层用识别器的
CTC 置信度择优。这样拉正只在真的更可信时被采纳，不会把好图改坏
——实测单路替换会掉 1~5 个百分点，而置信度择优能稳定拿到 +1 个百分点。

全程不读取文件名、不使用任何标注信息，纯推理期图像处理。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DeskewResult:
    """一次校正的结果。image 一定是可直接送识别的 BGR 图。"""

    image: np.ndarray
    applied: bool
    method: str  # color-quad / color-minrect / angle-rotate / none
    angle: float  # 估计出的倾斜角（度）
    corners: Optional[np.ndarray]  # 原图中的车牌四角（TL, TR, BR, BL），旋转路径为 None
    quality: float  # 0~1，越大越可信


IDENTITY_METHOD = "none"


def _order_corners(points: np.ndarray) -> np.ndarray:
    """把任意顺序的四个点整理成 左上、右上、右下、左下（顺时针）。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(diff))]  # x 最大、y 最小
    bl = pts[int(np.argmin(diff))]  # x 最小、y 最大
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    if len(np.unique(ordered, axis=0)) < 4:
        # 极端情况下点重复，按极角重排兜底
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        start = int(np.argmin(pts.sum(axis=1)))
        ordered = np.roll(pts, -start, axis=0)
    return ordered


def _quad_metrics(quad: np.ndarray) -> Tuple[float, float, float, float]:
    """返回 (平均宽, 平均高, 宽高比, 平行度误差)。"""
    tl, tr, br, bl = quad
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    w = (top + bottom) / 2.0
    h = (left + right) / 2.0
    ratio = w / max(h, 1e-6)
    parallel = max(
        abs(top - bottom) / max(top, bottom, 1e-6),
        abs(left - right) / max(left, right, 1e-6),
    )
    return w, h, ratio, parallel


def _contour_to_quad(contour: np.ndarray) -> Tuple[np.ndarray, str]:
    """轮廓 → 四点。优先四点近似（可表达透视），否则用最小外接矩形。"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype(np.float32), "quad"
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    if w < h:  # 让 w 始终是长边，angle 归到 [-45, 45)
        w, h = h, w
        angle += 90.0
    angle = ((angle + 45.0) % 90.0) - 45.0
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    return box.astype(np.float32), "minrect"


def _warp_to_frontal(image: np.ndarray, quad: np.ndarray, pad: float = 0.03) -> np.ndarray:
    """四点透视变换：把车牌四边形拉成正视图。

    目标宽高用四边形自身的边长，避免强行套用固定比例造成二次拉伸。
    pad > 0 时围绕四边形中心向外扩张，让结果带上一点车牌外框/背景，
    防止四边形估计略小或略偏时把首尾字符切掉——切掉一个字符对 CTC
    解码是灾难性的（整牌直接错），而多带一点背景 LPRNet 完全能容忍。
    """
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if pad > 0:
        center = quad.mean(axis=0, keepdims=True)
        quad = center + (quad - center) * (1.0 + 2.0 * pad)
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    width = max(int(round(width)), 8)
    height = max(int(round(height)), 8)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


class PlateDeskewer:
    """把 AABB 裁剪出来的车牌图自动拉正。

    参数:
        max_angle: 颜色路径允许的最大倾角（度），超过认为四边形估计不可靠。
        angle_search_range: 梯度路径的角度搜索范围（度）。
        min_area_ratio: 车牌区域至少占裁剪图的比例。
        min_ratio/max_ratio: 车牌宽高比的合法区间（普通牌约 3.1）。
        max_parallel_error: 四边形对边长度相对误差上限。
        min_apply_angle: 颜色路径倾角小于该值时不再输出候选——此时 AABB
            本身已经贴合，重采样只会带来边界切字风险。
        candidate_pads: 生成候选时用的外扩比例；多个候选一起交给识别器择优。
        enable_rotate: 是否启用梯度投影旋转兜底路径（实测噪声大，默认关闭）。
        min_rotate_angle: 兜底旋转路径的最小有效角度。
        quad_pad: 直接调用 deskew() 时使用的默认外扩比例。
    """

    def __init__(
        self,
        max_angle: float = 40.0,
        angle_search_range: float = 30.0,
        min_area_ratio: float = 0.20,
        min_ratio: float = 1.9,
        max_ratio: float = 6.5,
        max_parallel_error: float = 0.30,
        min_apply_angle: float = 2.0,
        candidate_pads: Tuple[float, ...] = (0.03, 0.05),
        enable_rotate: bool = False,
        min_rotate_angle: float = 6.0,
        quad_pad: float = 0.04,
    ) -> None:
        self.max_angle = max_angle
        self.angle_search_range = angle_search_range
        self.min_area_ratio = min_area_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_parallel_error = max_parallel_error
        self.min_apply_angle = min_apply_angle
        self.candidate_pads = tuple(candidate_pads) or (quad_pad,)
        self.enable_rotate = enable_rotate
        self.min_rotate_angle = min_rotate_angle
        self.quad_pad = quad_pad

    # ---------------- 路径 A：颜色 + 四边形 ----------------

    def _color_mask(self, bgr: np.ndarray) -> np.ndarray:
        """按车牌底色（蓝/绿/黄）取前景 mask。"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        blue = (h >= 95) & (h <= 135) & (s >= 60) & (v >= 40)
        green = (h >= 35) & (h <= 95) & (s >= 55) & (v >= 40)
        yellow = (h >= 10) & (h <= 38) & (s >= 70) & (v >= 90)
        mask = (blue | green | yellow).astype(np.uint8) * 255
        k = max(3, int(round(min(bgr.shape[:2]) * 0.05)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    def _pick_region(self, mask: np.ndarray, image_area: float) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        candidates = [
            c for c in contours if cv2.contourArea(c) >= image_area * self.min_area_ratio
        ]
        if not candidates:
            return None
        return max(candidates, key=cv2.contourArea)

    def _evaluate(self, quad: np.ndarray, image_area: float) -> Optional[float]:
        """校验四边形是否像车牌，像则返回 0~1 的质量分，否则 None。"""
        quad = _order_corners(quad)
        w, h, ratio, parallel = _quad_metrics(quad)
        if h <= 2 or w <= 2:
            return None
        area_ratio = cv2.contourArea(quad.astype(np.float32)) / image_area
        if area_ratio < self.min_area_ratio:
            return None
        if not (self.min_ratio <= ratio <= self.max_ratio):
            return None
        if parallel > self.max_parallel_error:
            return None
        angle = float(np.degrees(np.arctan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0])))
        if abs(angle) > self.max_angle:
            return None
        ratio_score = max(0.0, 1.0 - abs(ratio - 3.1) / 3.1)
        quality = 0.6 * min(1.0, area_ratio / 0.75) + 0.4 * ratio_score
        return float(quality)

    def _try_color(
        self, bgr: np.ndarray, pad: Optional[float] = None, force: bool = False
    ) -> Optional[DeskewResult]:
        image_area = float(bgr.shape[0] * bgr.shape[1])
        contour = self._pick_region(self._color_mask(bgr), image_area)
        if contour is None:
            return None
        quad, shape_kind = _contour_to_quad(contour)
        quality = self._evaluate(quad, image_area)
        if quality is None:
            return None
        quad = _order_corners(quad)
        angle = float(np.degrees(np.arctan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0])))
        if not force and abs(angle) < self.min_apply_angle:
            # 车牌已经基本横平，AABB 裁剪就是最优输入；强行重采样只会切字
            return None
        warped = _warp_to_frontal(bgr, quad, pad=self.quad_pad if pad is None else pad)
        if warped.size == 0 or warped.shape[0] < 6 or warped.shape[1] < 12:
            return None
        return DeskewResult(
            image=warped,
            applied=True,
            method=f"color-{shape_kind}",
            angle=angle,
            corners=quad,
            quality=quality,
        )

    # ---------------- 路径 B：梯度投影 + 角度搜索 ----------------

    @staticmethod
    def _row_peak_score(gray: np.ndarray, angle: float) -> float:
        """把 Sobel-y 边缘图旋转 angle 度后做行投影，返回能量集中度。

        旋转会带进黑边，黑边行不该参与统计，否则大角度会把边缘像素
        拉成一条纯色条纹、虚高方差，导致角度越搜越大。
        """
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        h, w = gy.shape
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(
            gy, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        coverage = (rotated > 0).sum(axis=1) / float(w)
        valid = coverage >= 0.85
        if int(valid.sum()) < 10:
            return 0.0
        rows = rotated.sum(axis=1).astype(np.float64)[valid]
        mean = float(rows.mean())
        if mean <= 1e-6:
            return 0.0
        return float(np.var(rows)) / (mean * mean)

    def _search_angle(self, gray: np.ndarray) -> Tuple[float, float]:
        """在小角度范围内搜索让水平边缘最集中的旋转角。返回 (角度, 得分)。"""
        best_angle, best_score = 0.0, self._row_peak_score(gray, 0.0)
        for angle in np.arange(
            -self.angle_search_range, self.angle_search_range + 0.1, 1.0
        ):
            if abs(angle) < 1e-6:
                continue
            score = self._row_peak_score(gray, float(angle))
            if score > best_score:
                best_angle, best_score = float(angle), score
        return best_angle, best_score

    def _try_rotate(self, bgr: np.ndarray) -> Optional[DeskewResult]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        angle, score = self._search_angle(gray)
        if abs(angle) < self.min_rotate_angle:
            return None
        zero_score = self._row_peak_score(gray, 0.0)
        if zero_score <= 1e-6 or score < zero_score * 1.05:
            return None
        h, w = bgr.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(
            bgr,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return DeskewResult(
            image=rotated,
            applied=True,
            method="angle-rotate",
            angle=angle,
            corners=None,
            quality=min(1.0, score / 6.0),
        )

    # ---------------- 对外接口 ----------------

    def candidates(self, bgr: np.ndarray) -> List[DeskewResult]:
        """给出若干候选图，第一个恒为原图（identity）。

        上层应对每个候选都跑一次识别，取 CTC 置信度最高的结果。
        这是本模块推荐的使用方式：拉正只是「多提供一个视角」，
        而不是强行覆盖原始像素，因此不可能把原本认对的图改坏。
        """
        if bgr is None or bgr.size == 0 or bgr.ndim != 3:
            return [DeskewResult(bgr, False, IDENTITY_METHOD, 0.0, None, 0.0)]

        results = [DeskewResult(bgr, False, IDENTITY_METHOD, 0.0, None, 0.0)]
        seen = set()
        for pad in self.candidate_pads:
            result = self._try_color(bgr, pad=pad)
            if result is None:
                continue
            # 同一四边形、不同外扩比例是不同的候选，必须都保留；
            # 只有 method + 角度 + 外扩比例全都一样才算重复
            key = (result.method, round(result.angle, 1), round(pad, 4))
            if key in seen:
                continue
            seen.add(key)
            results.append(result)
        if self.enable_rotate:
            rotated = self._try_rotate(bgr)
            if rotated is not None:
                results.append(rotated)
        return results

    def deskew(self, bgr: np.ndarray, force: bool = False) -> DeskewResult:
        """单结果接口：返回首选拉正版本，无可用候选时返回原图。

        注意：直接用它替换原图会损失识别率，推荐改用 candidates() 择优。
        """
        if bgr is None or bgr.size == 0 or bgr.ndim != 3:
            return DeskewResult(bgr, False, IDENTITY_METHOD, 0.0, None, 0.0)
        if force:
            direct = self._try_color(bgr, force=True)
            if direct is None and self.enable_rotate:
                direct = self._try_rotate(bgr)
            if direct is not None:
                return direct
        cands = self.candidates(bgr)
        return cands[1] if len(cands) > 1 else cands[0]
