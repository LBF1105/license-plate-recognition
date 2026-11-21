# License Plate Recognition

基于 Python 的车牌识别系统：采用 **YOLO 车牌检测 + LPRNet 字符识别** 的两阶段方案，支持对车辆图片进行车牌定位、号码识别与结果可视化。

> 多媒体技术实践课程项目

---

## 一、项目简介

本系统接收车辆图像作为输入，自动定位图像中的车牌区域，并进一步识别出完整的车牌号码，最终输出标注了车牌框与号码的结果图片。可应用于停车场道闸、小区门禁、高速收费、交通卡口等场景。

### 功能特性

- 车牌定位：基于 YOLO 检测模型输出车牌边界框
- 字符识别：基于 LPRNet（CTC）端到端输出车牌字符串，无需字符分割
- 结果可视化：在原图上绘制车牌框并标注识别号码
- 批量处理：支持对整个图片文件夹进行批量识别
- 支持普通蓝牌（7 位）与新能源绿牌（8 位）

---

## 二、技术架构

系统采用两阶段级联方案：

```
输入图片
   │
   ▼
[ YOLO 车牌检测 ]  ── 输出车牌边界框
   │
   ▼
[ 裁剪车牌区域 ]
   │
   ▼
[ LPRNet 字符识别 ]  ── CTC 解码输出字符串
   │
   ▼
[ 结果标注与保存 ]
```

| 阶段 | 模型 | 作用 |
| --- | --- | --- |
| 车牌检测 | YOLO（YOLOv8 / YOLOv5） | 在整图中定位车牌位置，输出边界框 |
| 字符识别 | LPRNet | 对裁剪出的车牌进行端到端字符识别 |

选择该方案的原因：车牌属于单一目标、特征明显，轻量 YOLO 即可快速准确检测；LPRNet 无需字符分割，模型仅几百 KB，CPU 上也能快速运行。

---

## 三、项目结构

```
license-plate-recognition/
├── README.md                    # 项目说明文档
├── LICENSE                      # 开源许可证（MIT）
├── .gitignore                   # Git 忽略规则
├── requirements.txt             # Python 依赖清单
├── config/
│   └── config.yaml              # 模型路径、阈值等配置
├── src/
│   └── plate_recognition/       # 核心源码包
│       ├── __init__.py
│       ├── detector.py          # YOLO 车牌检测模块
│       ├── recognizer.py        # LPRNet 字符识别模块
│       └── pipeline.py          # 检测 + 识别完整流程
├── scripts/                     # 可执行脚本
│   ├── infer_image.py           # 单图推理
│   ├── batch_test.py            # 批量测试与正确率统计
│   └── parse_ccpd.py            # CCPD 文件名真值解析
├── weights/                     # 模型权重（不进 git，见该目录 README）
├── data/
│   ├── test_images/             # 测试图片（不进 git）
│   └── README.md                # 数据集说明
├── output/                      # 识别结果输出（不进 git）
├── docs/                        # 补充文档
│   └── project_structure.md
└── tests/                       # 测试
    └── test_smoke.py
```

---

## 四、环境要求

- 操作系统：Windows / Linux / macOS
- Python：3.9 – 3.12
- 运行方式：CPU 或 GPU（NVIDIA，CUDA）均可

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

### 3. 准备模型权重

模型权重不包含在 Git 仓库中，请按照 [`weights/README.md`](weights/README.md) 的说明下载，并放置到 `weights/` 目录。

### 4. 准备测试图片

将待识别的车辆图片放入 `data/test_images/`，数据集说明见 [`data/README.md`](data/README.md)。

### 5. 运行识别

单张图片：

```bash
python scripts/infer_image.py --source data/test_images/test.jpg
```

批量识别整个文件夹：

```bash
python scripts/batch_test.py --source data/test_images
```

结果图片保存在 `output/` 目录。

---

## 六、数据集

本项目使用公开中文车牌数据集 **CCPD（Chinese City Parking Dataset）**，包含正常、模糊、旋转、倾斜、光照不均、雨雪天气等多种场景子集。下载与使用方式见 [`data/README.md`](data/README.md)。

---

## 七、测试

```bash
pytest tests/
```

---

## 八、团队分工

| 成员 | 主要职责 |
| --- | --- |
| 成员 A | 环境搭建、核心识别模块、代码整合 |
| 成员 B | 数据集准备、真值解析 |
| 成员 C | 批量测试、正确率统计、结果分析 |
| 成员 D | 可视化、流程图、文档与 PPT |

每位成员均须参与程序运行与实验测试，每人至少完成 5 张图片的识别测试。

---

## 九、许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 十、参考资料与致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [CCPD 数据集](https://github.com/detectRecog/CCPD)
- LPRNet: License Plate Recognition via Deep Neural Networks

本项目在环境配置、代码理解与调试过程中使用了 AI 工具辅助，所有实验结果与数据均来自实际运行。
