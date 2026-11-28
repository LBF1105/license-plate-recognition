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
| `plate_detect.pt` | YOLOv8（we0091234/yolov8-plate） | 车牌检测，输出边界框（单层/双层两类） |
| `lprnet.pth` | LPRNet | 车牌字符识别，输出号码 |
| `corner_net.pt` | CornerNet（本项目训练） | 车牌四角点回归，用于倾斜车牌透视拉正，约 2 MB |

### 获取方式

1. 从开源中文车牌识别项目 we0091234/yolov8-plate 下载预训练权重（基于 CCPD 训练），直达链接见下方。
2. 下载后按上表重命名并放入本目录。
3. 也可使用 CCPD 自行训练后将权重保存到本目录。
plate_detect.pt 获取网址：https://raw.githubusercontent.com/we0091234/yolov8-plate/master/weights/yolov8s.pt
lprnet.pth 获取网址：https://raw.githubusercontent.com/we0091234/yolov8-plate/master/weights/plate_rec_color.pth

下载后需改名放入本目录：`yolov8s.pt` → `plate_detect.pt`，`plate_rec_color.pth` → `lprnet.pth`。

### corner_net.pt（本项目自行训练，随仓库提供）

`corner_net.pt` 是本项目自己训练的角点回归网络，用来把倾斜车牌按四个真实顶点拉正。
它**不同于上面两个下载的权重，直接随仓库提交**（体积仅约 2 MB），clone 下来即可用。

如需重新训练：

```bash
python scripts/train_corner_net.py --source data/test_images --epochs 200 --rebuild
```

该权重缺失时程序不会崩溃：会自动打印提示并降级为纯几何兜底（`deskewer.py`），
再用 `--no-corner` 可以完全关闭角点拉正。详见 [../docs/corner_net.md](../docs/corner_net.md)。

## 放置后的目录结构

```
weights/
├── README.md              # 本说明
├── plate_detect.pt        # 需自行下载
├── lprnet.pth             # 需自行下载
└── corner_net.pt          # 本项目训练，随仓库提供
```

## 配置

权重路径在 `config/config.yaml` 的 `two_stage` 段配置，如文件名不同，请同步修改 `detector.weights` 与 `recognizer.weights`。
