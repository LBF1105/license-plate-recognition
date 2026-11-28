"""训练车牌四角点回归网络（CornerNet）。

数据：CCPD 文件名第 4 段自带四角点真值，无需额外标注。
输入：YOLO 检测框按 CROP_MARGIN 外扩后的裁剪图
标签：四角点换算到裁剪图坐标后归一化到 [0,1]

用法：
    # 全量训练，产出部署用的 weights/corner_net.pt
    python scripts/train_corner_net.py --final

    # 5 折交叉验证评估（OOF），输出角点误差与端到端识别率
    python scripts/train_corner_net.py --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from plate_recognition.two_stage.corner_net import (INPUT_H, INPUT_W, CornerNet,
                                                   order_quad)
from plate_recognition.two_stage.detector import PlateDetector

MARGIN = 0.05
SEED = 20241128


# ---------------------------------------------------------------- 数据集构建

def parse_corners(name: str):
    """CCPD 文件名第 4 段：x1&y1_x2&y2_x3&y3_x4&y4"""
    parts = name.split("-")
    if len(parts) < 5:
        return None
    try:
        pts = [tuple(float(v) for v in seg.split("&")) for seg in parts[3].split("_")]
    except Exception:
        return None
    if len(pts) != 4:
        return None
    return order_quad(np.array(pts, dtype=np.float32))


def build_cache(source: Path, cache: Path, detector_weights: str) -> list[dict]:
    """检测 + 裁剪 + 角点换算，结果缓存在 cache 目录。"""
    from parse_ccpd import parse_plate

    sys.path.insert(0, str(ROOT / "scripts"))
    from parse_ccpd import parse_plate  # noqa: F811

    (cache / "crops").mkdir(parents=True, exist_ok=True)
    det = PlateDetector(detector_weights, 0.25, 0.45, "cpu")
    det.load()

    meta, skip = [], {"no_gt": 0, "no_det": 0, "double": 0, "bad_corner": 0}
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".png"})
    for p in images:
        try:
            gt = parse_plate(p.name)
            tilt = int(p.name.split("-")[1].split("_")[0]) - 90
        except Exception:
            skip["no_gt"] += 1
            continue
        quad = parse_corners(p.name)
        if quad is None:
            skip["bad_corner"] += 1
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            skip["no_gt"] += 1
            continue
        dets = det.detect_detailed(img)
        if not dets:
            skip["no_det"] += 1
            continue
        x1, y1, x2, y2, _conf, cls_id = dets[0]
        if cls_id == 1:
            skip["double"] += 1
            continue
        mx, my = int((x2 - x1) * MARGIN), int((y2 - y1) * MARGIN)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(img.shape[1], x2 + mx), min(img.shape[0], y2 + my)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            skip["no_det"] += 1
            continue
        ch, cw = crop.shape[:2]
        cv2.imwrite(str(cache / "crops" / f"{p.stem}.png"), crop)
        meta.append({
            "name": p.stem,
            "file": p.name,
            "crop": f"crops/{p.stem}.png",
            "w": int(cw), "h": int(ch),
            "corners": (quad - np.array([cx1, cy1], np.float32)).tolist(),
            "gt": gt,
            "tilt": tilt,
        })

    (cache / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"数据集: {len(meta)} 张  (跳过 {skip})")
    return meta


# ---------------------------------------------------------------- 训练用 dataset

class CornerDS(Dataset):
    def __init__(self, items, cache: Path, train: bool, seed: int = SEED):
        self.items, self.cache, self.train = items, cache, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        crop = cv2.imread(str(self.cache / it["crop"]), cv2.IMREAD_COLOR)
        h, w = crop.shape[:2]
        corners = np.array(it["corners"], dtype=np.float32)
        if self.train:
            crop, corners = self._augment(crop, corners)
        norm = np.stack([corners[:, 0] / w, corners[:, 1] / h], axis=1).reshape(-1)
        return _to_input(crop), torch.from_numpy(norm.astype(np.float32))

    def _augment(self, crop, corners):
        h, w = crop.shape[:2]
        rng = self.rng
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rng.uniform(-9, 9),
                                    rng.uniform(0.88, 1.12))
        m[0, 2] += rng.uniform(-0.06, 0.06) * w
        m[1, 2] += rng.uniform(-0.06, 0.06) * h
        img = cv2.warpAffine(crop, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        pts = corners @ m[:, :2].T + m[:, 2]
        img = np.clip(img.astype(np.float32) * rng.uniform(0.78, 1.22) + rng.uniform(-28, 28),
                      0, 255).astype(np.uint8)
        if rng.random() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img, pts


def _to_input(crop):
    r = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(r[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5


# ---------------------------------------------------------------- 训练

def train(items, cache, epochs, device, seed=SEED, log_prefix=""):
    ds = CornerDS(items, cache, True, seed)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    model = CornerNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss(beta=0.05)
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"  {log_prefix}epoch {ep+1}/{epochs}  loss {total/n:.5f}")
    return model


def main():
    ap = argparse.ArgumentParser(description="训练角点回归网络")
    ap.add_argument("--source", default=str(ROOT / "data/test_images"))
    ap.add_argument("--detector", default=str(ROOT / "weights/plate_detect.pt"))
    ap.add_argument("--cache", default="/tmp/corner_ds")
    ap.add_argument("--out", default=str(ROOT / "weights/corner_net.pt"))
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--folds", type=int, default=0, help=">0 时做交叉验证评估，不保存模型")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rebuild", action="store_true", help="强制重建缓存")
    args = ap.parse_args()

    cache = Path(args.cache)
    meta_path = cache / "meta.json"
    if args.rebuild or not meta_path.exists():
        meta = build_cache(Path(args.source), cache, args.detector)
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"数据集(缓存): {len(meta)} 张")

    print(f"device = {args.device}")

    if args.folds <= 1:
        model = train(meta, cache, args.epochs, args.device, log_prefix="")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"模型已保存: {args.out}")
        return

    # ---- 交叉验证 ----
    from plate_recognition.two_stage.corner_net import CornerPredictor
    from plate_recognition.two_stage.recognizer import PlateRecognizer

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(meta))
    folds = np.array_split(order, args.folds)

    oof = [None] * len(meta)
    for f in range(args.folds):
        val = set(folds[f].tolist())
        train_items = [meta[i] for i in range(len(meta)) if i not in val]
        model = train(train_items, cache, args.epochs, args.device, seed=SEED + f,
                      log_prefix=f"fold{f} ")
        pred = CornerPredictor("", args.device)
        pred.model = model.eval()
        for i in folds[f]:
            it = meta[i]
            crop = cv2.imread(str(cache / it["crop"]), cv2.IMREAD_COLOR)
            oof[i] = pred.predict(crop).corners
        print(f"  fold{f}: val {len(val)} 张完成")

    errs = np.array([
        np.linalg.norm(oof[i] - np.array(meta[i]["corners"], np.float32), axis=1).mean()
        / meta[i]["w"] * 100 for i in range(len(meta))
    ])
    print(f"\n角点误差（相对车牌宽度）: 中位 {np.median(errs):.2f}%  均值 {errs.mean():.2f}%  "
          f"<3% {100*(errs < 3).mean():.0f}%  <5% {100*(errs < 5).mean():.0f}%")

    rec = PlateRecognizer(str(ROOT / "weights/lprnet.pth"), "cpu")
    rec.load()
    rows = []
    for i, it in enumerate(meta):
        crop = cv2.imread(str(cache / it["crop"]), cv2.IMREAD_COLOR)
        b_text, b_score = rec.recognize_with_score(crop)
        wim = CornerPredictor.warp(crop, oof[i])
        c_text, c_score = rec.recognize_with_score(wim) if wim is not None else (b_text, b_score)
        best = c_text if (wim is not None and c_score > b_score) else b_text
        rows.append({"gt": it["gt"], "base": b_text, "corner": c_text, "best": best,
                     "tilt": it["tilt"]})

    n = len(rows)
    b = sum(r["base"] == r["gt"] for r in rows)
    c = sum(r["corner"] == r["gt"] for r in rows)
    s = sum(r["best"] == r["gt"] for r in rows)
    print(f"\n端到端（{n} 张 OOF）:")
    print(f"  baseline 原图         : {b}/{n} = {100*b/n:.1f}%")
    print(f"  角点拉正(直接替换)    : {c}/{n} = {100*c/n:.1f}%")
    print(f"  角点拉正 + 择优       : {s}/{n} = {100*s/n:.1f}%")
    for lo, hi, lab in ((0, 5, "≤5°"), (5, 15, "6-15°"), (15, 90, ">15°")):
        sub = [r for r in rows if lo <= abs(r["tilt"]) < hi]
        if not sub:
            continue
        bb = sum(r["base"] == r["gt"] for r in sub)
        cc = sum(r["corner"] == r["gt"] for r in sub)
        ss = sum(r["best"] == r["gt"] for r in sub)
        print(f"  {lab:<6} {len(sub):3d} 张: baseline {100*bb/len(sub):5.1f}%  "
              f"拉正 {100*cc/len(sub):5.1f}%  择优 {100*ss/len(sub):5.1f}%")


if __name__ == "__main__":
    main()
