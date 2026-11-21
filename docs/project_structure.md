# 项目结构说明

本文档描述双方案对照系统的目录组织、各模块职责与数据流转方式。

## 目录结构

```
license-plate-recognition/
├── README.md                    # 项目主文档
├── LICENSE                      # MIT 许可证
├── .gitignore                   # Git 忽略规则
├── requirements.txt             # 依赖清单
│
├── config/
│   └── config.yaml              # 双方案配置：档位、权重、阈值、设备
│
├── src/plate_recognition/       # 核心源码包
│   ├── __init__.py
│   ├── integrated/              # 方案一：集成式
│   │   ├── __init__.py
│   │   └── hyperlpr_runner.py   # IntegratedRecognizer
│   ├── two_stage/               # 方案二：分步式
│   │   ├── __init__.py
│   │   ├── detector.py          # PlateDetector（YOLO）
│   │   ├── recognizer.py        # PlateRecognizer（LPRNet）
│   │   └── pipeline.py          # TwoStagePipeline
│   └── common/                  # 共用工具
│       ├── __init__.py
│       ├── image_io.py          # 图像读取与列举
│       ├── visualize.py         # 画框标注
│       └── metrics.py           # 正确率与耗时
│
├── scripts/                     # 可执行入口
│   ├── run_integrated.py        # 运行集成式方案
│   ├── run_two_stage.py         # 运行分步式方案
│   ├── compare.py               # 双方案对照
│   └── parse_ccpd.py            # CCPD 真值解析
│
├── weights/                     # 模型权重（不进 git）
├── data/
│   ├── test_images/             # 测试图片（不进 git）
│   └── README.md
├── output/                      # 分方案输出（不进 git）
│   ├── integrated/
│   ├── two_stage/
│   └── comparison/
│
├── docs/
│   ├── project_structure.md     # 本文档
│   └── experiment_design.md     # 对照实验设计
└── tests/
    └── test_smoke.py
```

## 模块职责

- **IntegratedRecognizer（integrated/hyperlpr_runner.py）**：封装 HyperLPR3，框架内部完成检测与识别，一次调用即返回车牌号码与位置。
- **PlateDetector（two_stage/detector.py）**：加载 YOLO 权重，返回车牌边界框与置信度。
- **PlateRecognizer（two_stage/recognizer.py）**：加载 LPRNet 权重，对裁剪车牌预处理、推理并 CTC 解码。
- **TwoStagePipeline（two_stage/pipeline.py）**：串联检测与识别，完成"检测 → 裁剪 → 识别"。
- **common/**：两方案共用的图像 IO、可视化与评估指标，保证对照口径一致。

## 数据流向

```
原图
 ├─ 方案一：IntegratedRecognizer.recognize() 内部完成检测+识别
 └─ 方案二：PlateDetector.detect() → 裁剪 → PlateRecognizer.recognize()
                                │
                                ▼
                    common.visualize 画框标注
                                │
                                ▼
              分别保存到 output/integrated 或 output/two_stage
```

## 配置加载

脚本统一读取 `config/config.yaml`，两方案的档位、权重路径、阈值与设备均在配置中声明，修改配置即可切换，无需改动源码。
