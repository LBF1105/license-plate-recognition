# HyperLPR3 集成式基线：环境搭建与实测记录

## 1. 目的

方案一（基础任务）：直接调用 HyperLPR3 预训练模型完成端到端车牌识别，
不训练、不调参，验证环境可用并记录 CPU 推理性能，作为后续
"图像预处理对照"和"轻量模型微调"两条路线的对照基准。

## 2. 环境搭建（WSL2 + Ubuntu）

```bash
# 1) 装系统级 pip 与 venv（Ubuntu 24.04+ 的 Python 受 PEP 668 保护，不能直接 pip install）
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 2) 在项目里建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3) 安装依赖（国内可用清华源加速）
pip install hyperlpr3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

- 首次 `import hyperlpr3` 会自动下载模型到 `~/.hyperlpr3/20230229/onnx/`，
  共 4 个 onnx 文件、约 11.9 MB，之后离线运行。
- 实测版本：hyperlpr3 0.1.3、numpy 2.5.5、opencv-python 5.1.0.95、onnxruntime 1.32.0
  （均提供 cp314/abi3 wheel，Python 3.14 可用）。

## 3. 运行方式

```bash
# 整图直接识别（默认）
python scripts/run_integrated.py --source data/test_images/demo2.jpg

# 分块识别（大图中远处小车牌漏检时使用）
python scripts/run_integrated.py --source data/test_images/demo2.jpg --tile
```

结果图默认写到 `output/integrated/<图片名>_result.jpg`，终端同时打印
每张车牌号码、置信度、位置与平均耗时。

## 4. 实测结果（demo2.jpg，4608×3456，CPU）

| 模式 | 识别数量 | 识别到的车牌 | 单张耗时 |
| --- | --- | --- | --- |
| low（320×320） | 2 | 冀A9F3J6、冀A7F691 | 约 0.36 s |
| high（640×640） | 3 | 冀A9F3J6、冀A8C309、苏E67A12 | 约 0.36 s |
| high + 分块 3×2 | 6 | 冀A8C309、冀A7F691、冀A7837P、冀A2036Y、苏E67A12、冀A9F3J6 | 约 1.1 s |

结论：

1. 集成式方案开箱即用，CPU 即可，适合作为基础任务交付。
2. low 与 high 漏检的车牌并不相同，高分辨率模式更稳，但都不是"全识别"。
3. 大图整体缩放到 320/640 后，远处车牌只剩几个像素，必然漏检；
   分块后每块内的车牌相对变大，召回从 2～3 个提升到 6 个。
4. 检测阶段默认 `box_threshold=0.5`、`nms_threshold=0.6`，
   置信度低的框会被直接过滤；识别阶段还会丢弃空串/长度不足的结果。

## 5. 已知限制

- 极远处、倾斜、遮挡的小车牌仍可能漏检（分块也无法解决）。
- 中文车牌无法用 OpenCV `putText` 直接绘制，`common/visualize.py`
  里做了省份简称→拼音的映射，标签显示为 `JiA9F3J6 1.00` 形式。
- 分块会带来重复框，已用 IoU>0.5 合并、保留置信度更高的结果。

## 6. 代码位置

- `src/plate_recognition/integrated/hyperlpr_runner.py`：识别器封装（整图/分块）
- `scripts/run_integrated.py`：命令行入口
- `src/plate_recognition/common/visualize.py`：结果可视化

## 7. 数据说明

`data/test_images/` 下的示例图被 `.gitignore` 忽略，需自行放入；
运行前把 `demo2.jpg` 放到该目录。
