# 项目结构说明

本文档描述系统的目录组织、各模块职责以及数据在模块间的流转方式。

## 目录结构

```
license-plate-recognition/
├── README.md                    # 项目主文档
├── LICENSE                      # MIT 许可证
├── .gitignore                   # Git 忽略规则
├── requirements.txt             # 依赖清单
│
├── config/
│   └── config.yaml              # 全局配置：权重路径、阈值、设备
│
├── src/plate_recognition/       # 核心源码包
│   ├── __init__.py              # 包声明与版本号
│   ├── detector.py              # PlateDetector：YOLO 车牌检测
│   ├── recognizer.py            # PlateRecognizer：LPRNet 字符识别
│   └── pipeline.py              # PlateRecognitionPipeline：流程串联与可视化
│
├── scripts/                     # 面向使用的可执行脚本
│   ├── infer_image.py           # 单张图片推理
│   ├── batch_test.py            # 批量识别与正确率统计
│   └── parse_ccpd.py            # CCPD 文件名真值解析
│
├── weights/                     # 模型权重（不进 git）
├── data/
│   ├── test_images/             # 测试图片（不进 git）
│   └── README.md                # 数据集说明
├── output/                      # 识别结果输出（不进 git）
│
├── docs/
│   └── project_structure.md     # 本文档
└── tests/
    └── test_smoke.py            # 冒烟测试
```

## 模块职责

- **PlateDetector（detector.py）**：加载 YOLO 权重，对输入图像返回车牌边界框列表，每个框包含坐标与置信度。
- **PlateRecognizer（recognizer.py）**：加载 LPRNet 权重，对裁剪出的车牌图像进行预处理、推理与 CTC 解码，返回车牌字符串。
- **PlateRecognitionPipeline（pipeline.py）**：组合检测器与识别器，完成"检测 → 裁剪 → 识别"，并提供画框标注方法。

## 数据流向

```
cv2.imread 读取原图
        │
        ▼
PlateDetector.detect()  →  [(x1, y1, x2, y2, conf), ...]
        │
        ▼
按框裁剪车牌区域 image[y1:y2, x1:x2]
        │
        ▼
PlateRecognizer.recognize()  →  车牌字符串
        │
        ▼
PlateRecognitionPipeline.draw()  →  标注图像
        │
        ▼
cv2.imwrite 保存到 output/
```

## 配置加载

脚本通过读取 `config/config.yaml` 获取权重路径、置信度阈值与运行设备，避免在代码中硬编码。修改配置后无需改动源码即可切换模型或运行设备。
