# 模型权重说明

本目录用于存放车牌识别系统所需的模型权重。**权重文件不纳入 Git 版本管理**（`.gitignore` 已忽略 `*.pt`、`*.pth`、`*.onnx`），原因是文件体积较大，且 GitHub 对单文件有 100 MB 的大小限制。

## 所需权重

| 文件名 | 模型 | 用途 |
| --- | --- | --- |
| `plate_detect.pt` | YOLO（YOLOv8 / YOLOv5） | 车牌检测，输出车牌边界框 |
| `lprnet.pth` | LPRNet | 车牌字符识别，输出号码字符串 |

## 获取方式

1. 从开源中文车牌识别项目（YOLO + LPRNet）的 README 中提供的网盘链接下载预训练权重，这类权重通常基于 CCPD 数据集训练。
2. 下载后按下表重命名并放入本目录。
3. 若自行训练，可使用 CCPD 数据集分别训练检测模型与识别模型，将产出的权重保存到本目录。

## 放置后的目录结构

```
weights/
├── README.md            # 本说明文档
├── plate_detect.pt      # 需自行下载
└── lprnet.pth           # 需自行下载
```

## 配置

权重路径在 `config/config.yaml` 中配置，如文件名与上表不同，请同步修改该配置文件中的 `detector.weights` 与 `recognizer.weights` 字段。
