# License Plate Recognition

基于 Python 的车牌识别双方案对照系统：以 **HyperLPR3 集成式识别**作为基础模型，以 **YOLO 车牌检测 + LPRNet 字符识别的分步式模型**作为对照，在同一组图片上比较整牌识别正确率、耗时与失败案例。

> 多媒体技术实践课程项目

---

## 一、项目简介

系统接收车辆图像作为输入，自动定位车牌区域并识别完整车牌号码，输出标注车牌框与号码的结果图片，可应用于停车场道闸、小区门禁、高速收费、交通卡口等场景。

本项目实现并对照两种方案：

| 方案 | 架构 | 特征 |
| --- | --- | --- |
| 方案一（基线） | HyperLPR3 集成式 | 框架内部端到端完成检测与识别，一次调用直接输出号码；**已实现并完成 CPU 实测** |
| 方案二（对照） | YOLO + LPRNet 分步式 | 检测与识别拆分为独立模块，可分别替换与调优 |

---

## 二、技术架构

### 方案一：集成式

```
输入图片
   │
   ▼
[ HyperLPR3 内部完成检测 + 识别 ]
   │
   ▼
车牌号码 + 位置
```

### 方案二：分步式

```
输入图片
   │
   ▼
[ YOLO 车牌检测 ] ── 边界框
   │
   ▼
[ 裁剪车牌区域 ]
   │
   ▼
[ LPRNet 字符识别 ] ── CTC 解码
   │
   ▼
车牌号码 + 位置
```

两方案使用统一的可视化与评估口径（`common` 模块），保证对照公平。

---

## 三、项目结构

```
license-plate-recognition/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config/config.yaml              # 双方案配置
├── src/plate_recognition/
│   ├── integrated/                 # 方案一：HyperLPR3
│   ├── two_stage/                  # 方案二：YOLO + LPRNet
│   └── common/                     # 共用 IO、可视化、指标
├── scripts/
│   ├── run_integrated.py           # 运行集成式
│   ├── run_two_stage.py            # 运行分步式
│   ├── compare.py                  # 双方案对照
│   └── parse_ccpd.py               # CCPD 真值解析
├── weights/                        # 权重（不进 git）
├── data/test_images/               # 测试图片（不进 git）
├── output/
│   ├── integrated/
│   ├── two_stage/
│   └── comparison/
├── docs/
│   ├── project_structure.md
│   ├── experiment_design.md
│   └── hyperlpr3_baseline.md       # 方案一环境搭建与实测记录
└── tests/
    ├── test_smoke.py
    └── test_integrated.py
```

---

## 四、环境要求

- 操作系统：Windows / Linux / macOS
- Python：3.9 – 3.14（已在 Python 3.14 + WSL2 Ubuntu 上验证）
- CPU 或 GPU（NVIDIA CUDA）均可，无需重新训练即可推理

---

## 五、快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/LBF1105/license-plate-recognition.git
cd license-plate-recognition
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> Ubuntu 24.04+ 的 Python 受 PEP 668 保护，建议先建虚拟环境：
> `python3 -m venv .venv && source .venv/bin/activate`。

### 3. 准备权重与测试图片

- 权重：见 [`weights/README.md`](weights/README.md)，HyperLPR3 随包加载，分步式需手动下载
- 测试图片：放入 `data/test_images/`，数据集说明见 [`data/README.md`](data/README.md)

### 4. 运行方案

```bash
# 方案一：集成式
python scripts/run_integrated.py --source data/test_images
# 方案一（分块检测，提升大图中远处小车牌召回）
python scripts/run_integrated.py --source data/test_images --tile
# 方案二：分步式
python scripts/run_two_stage.py --source data/test_images
```

### 5. 双方案对照

```bash
python scripts/parse_ccpd.py --source data/test_images
python scripts/compare.py --source data/test_images --ground-truth output/ground_truth.csv
```

对照结果输出到 `output/comparison`，实验设计见 [`docs/experiment_design.md`](docs/experiment_design.md)。

---

## 六、方案一实测结果（CPU）

测试图 `demo2.jpg`（4608×3456，小区停车场景，可见 6 个车牌）：

| 模式 | 识别数量 | 识别到的车牌 |
| --- | --- | --- |
| low（320×320） | 2 | 冀A9F3J6、冀A7F691 |
| high（640×640） | 3 | 冀A9F3J6、冀A8C309、苏E67A12 |
| high + 分块 3×2 | 6 | 冀A8C309、冀A7F691、冀A7837P、冀A2036Y、苏E67A12、冀A9F3J6 |

单张 CPU 推理约 0.36 s（整图）/ 1.1 s（分块）。环境搭建步骤、参数影响与失败案例分析见
[`docs/hyperlpr3_baseline.md`](docs/hyperlpr3_baseline.md)。

---

## 七、数据集

使用公开中文车牌数据集 **CCPD（Chinese City Parking Dataset）**，包含正常、模糊、旋转、倾斜、光照不均、雨雪天气等场景子集，详见 [`data/README.md`](data/README.md)。

---

## 八、测试

```bash
pytest tests/
```

---

## 九、团队分工

| 成员 | 主要职责 |
| --- | --- |
| 成员 A | 环境搭建、双方案核心模块、代码整合 |
| 成员 B | 数据集准备、CCPD 真值解析 |
| 成员 C | 批量测试、正确率统计、对照与结果分析 |
| 成员 D | 可视化、流程图、文档与 PPT |

每位成员均须参与程序运行与实验测试，每人至少完成 5 张图片的识别测试。

---

## 十、许可证

基于 [MIT License](LICENSE) 开源。

---

## 十一、参考资料

- [HyperLPR3](https://github.com/szad670401/HyperLPR)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [CCPD 数据集](https://github.com/detectRecog/CCPD)
- LPRNet: License Plate Recognition via Deep Neural Networks

本项目在环境配置、代码理解与调试过程中使用了 AI 工具辅助，所有实验结果与数据均来自实际运行。
