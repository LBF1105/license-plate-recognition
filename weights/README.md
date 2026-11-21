# 模型权重说明

模型权重**不纳入 Git 版本管理**（`.gitignore` 已忽略 `*.pt`、`*.pth`、`*.onnx`），原因是体积较大，且 GitHub 单文件有 100 MB 限制。

## 方案一：集成式 HyperLPR3

HyperLPR3 通过 pip 安装，其模型权重随软件包内置或在首次运行时自动加载，**一般无需手动下载权重文件**。

```bash
pip install hyperlpr3
```

## 方案二：分步式 YOLO + LPRNet

需要手动准备以下两个权重：

| 文件名 | 模型 | 用途 |
| --- | --- | --- |
| `plate_detect.pt` | YOLO（YOLOv8 / YOLOv5） | 车牌检测，输出边界框 |
| `lprnet.pth` | LPRNet | 车牌字符识别，输出号码 |

### 获取方式

1. 从开源中文车牌识别项目（YOLO + LPRNet）README 提供的网盘链接下载预训练权重，通常基于 CCPD 训练。
2. 下载后按上表重命名并放入本目录。
3. 也可使用 CCPD 自行训练后将权重保存到本目录。
plate_detect.pt获取网址：https://raw.githubusercontent.com/we0091234/yolov8-plate/master/weights/yolov8s.pt
lprnet.pth获取网址：https://raw.githubusercontent.com/we0091234/yolov8-plate/master/weights/plate_rec_color.pth

## 放置后的目录结构

```
weights/
├── README.md              # 本说明
├── plate_detect.pt        # 需自行下载
└── lprnet.pth             # 需自行下载
```

## 配置

权重路径在 `config/config.yaml` 的 `two_stage` 段配置，如文件名不同，请同步修改 `detector.weights` 与 `recognizer.weights`。
