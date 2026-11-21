# integrated：HyperLPR3 集成式识别

本模块封装 [HyperLPR3](https://github.com/szad670401/HyperLPR)，由框架内部完成
"车牌检测 + 字符识别"，调用一次即可拿到整张图片的全部车牌号码与位置。

```
输入图片 ──► HyperLPR3 LicensePlateCatcher ──► [号码, 置信度, 类型, 框]
```

## 文件

| 文件 | 说明 |
| --- | --- |
| `hyperlpr_runner.py` | `IntegratedRecognizer`，唯一的对外入口 |
| `__init__.py` | 包声明 |

## 快速使用

```python
import cv2

from plate_recognition.integrated.hyperlpr_runner import IntegratedRecognizer

# 1) 创建并加载识别器（首次会自动下载模型，见下文）
recognizer = IntegratedRecognizer(detect_level="high", tile=True)
recognizer.load()

# 2) 整张图直接识别
image = cv2.imread("samples/demo2.jpg")
results = recognizer.recognize(image)

# 3) 结果：[{"plate": "冀A9F3J6", "confidence": 0.9999, "bbox": [x1, y1, x2, y2]}, ...]
for item in results:
    print(item["plate"], f"{item['confidence']:.4f}", item["bbox"])
```

不想写代码时用命令行入口（支持单图、目录批量、多进程、CSV 汇总）：

```bash
python scripts/run_integrated.py --source samples/demo2.jpg --tile
python scripts/run_integrated.py --source data/test_images --tile --workers 4 --skip-existing
```

## 构造参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `detect_level` | `"high"` | 检测档位：`"high"`（640×640）或 `"low"`（320×320）。高分辨率对远处小车牌更稳，但更慢 |
| `device` | `"cpu"` | 计算设备。当前 HyperLPR3 封装走 onnxruntime，CPU 即可运行 |
| `tile` | `False` | 是否分块检测。大图整体缩放后远处车牌会漏检，开启后切块逐块识别 |
| `tile_grid` | `(3, 2)` | 分块网格（横向块数, 纵向块数） |
| `overlap` | `0.15` | 分块重叠比例，避免车牌正好被切在边界 |
| `iou_threshold` | `0.5` | 去重阈值：两个框 IoU 大于该值视为同一车牌，保留置信度高的 |

## 返回结构

```python
[
    {
        "plate": str,            # 车牌号码，如 "冀A9F3J6"
        "confidence": float,     # 置信度 0～1
        "bbox": [x1, y1, x2, y2] # 左上角、右下角像素坐标，基于输入图坐标系
    },
    ...
]
```

没有检测到车牌时返回空列表。

## 两种识别模式

| 模式 | 原理 | 耗时（CPU，4608×3456） | 适用 |
| --- | --- | --- | --- |
| 整图 | 原图缩放到 320/640 后一次推理 | 约 0.4 s | 图片本身不大，或车牌都比较大 |
| 分块 `tile=True` | 切成 3×2（带 15% 重叠），逐块推理，坐标回填后按 IoU 去重 | 约 0.9～1.2 s | 大图、车牌多、远处小车牌漏检 |

实测 `samples/demo2.jpg`：整图 high 识别 3 个，分块识别 6 个（详见
`docs/hyperlpr3_baseline.md`）。

> 分块的代价是同一车牌可能被相邻两块各检出一次，`_dedup()` 会按 IoU>0.5
> 合并并保留置信度更高的结果；块之间的重叠区域是刻意保留的，不要设为 0。

## 模型文件

首次 `load()` 时自动从 CDN 下载到 `~/.hyperlpr3/20230229/onnx/`，约 11.9 MB
（4 个 onnx），之后可离线运行。没有网络的环境需要提前把该目录拷贝到目标机器同名路径。

## 依赖

`hyperlpr3`、`onnxruntime`、`opencv-python`、`numpy`（见仓库根目录
`requirements-basic.txt`）。安装：

```bash
pip install -r requirements-basic.txt
```

## 注意事项

- **中文车牌绘制**：OpenCV 的 `putText` 不支持中文，可视化请用
  `plate_recognition.common.visualize.draw_results()`，它会把省份简称转成拼音
  （`冀A9F3J6` → `JiA9F3J6`）。
- **坐标是原图坐标**：分块模式下已把框映射回整图坐标系，可直接用于画框。
- **检测阈值**：如需更高召回，可以在 `load()` 后调整
  `recognizer.catcher.pipeline.detector.box_threshold`（默认 0.5，调低误检也会变多）。
- 异常处理：`load()` 前调用 `recognize()` 会抛 `RuntimeError`；图片读不出由调用方
  负责（命令行入口会记录为失败并继续）。
