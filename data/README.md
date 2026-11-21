# 数据集说明

## 数据集：CCPD（Chinese City Parking Dataset）

本项目使用中科大团队发布的公开中文车牌数据集 CCPD，采集自城市停车场，包含不同光照、角度、距离、天气与模糊程度的真实车牌图像，并带有车牌位置与字符标注。

- 官方仓库：<https://github.com/detectRecog/CCPD>
- 许可：仅供研究与教学使用

## 子集说明

| 子集 | 场景 |
| --- | --- |
| `ccpd_base` | 正常车牌 |
| `ccpd_blur` | 拍摄模糊 |
| `ccpd_rotate` | 水平旋转 |
| `ccpd_tilt` | 倾斜 |
| `ccpd_db` | 光照过亮、过暗或不均 |
| `ccpd_weather` | 雨、雪、雾天气 |
| `ccpd_challenge` | 综合困难样本 |
| `ccpd_np` | 未悬挂车牌的新车 |

## 下载方式

- 百度网盘（2019 蓝牌版，提取码 `hm0u`）、Google Drive，链接见官方仓库 README。
- 新能源绿牌使用 CCPD2020（CCPD-Green，8 位字符，提取码 `ol3j`）。

## 仓库自带的 300 张样本

仓库的 `data/test_images/` 随仓库提供 300 张从 CCPD2019 训练集随机抽取的图片
（另含 1 张 `samples/demo2.jpg` 的副本，共 301 张，约 27 MB），克隆后即可直接运行。

- **来源**：CCPD2019 训练集（10 万张）的 HuggingFace 镜像
  <https://huggingface.co/datasets/okita-souji/ccpd2019train>，原始数据以官方仓库链接为准。
- **抽取方式**：训练集压缩包约 5.4 GB，不做整包下载；zip 支持随机访问，用 `remotezip`
  按 HTTP Range 只读取中央目录和随机 300 个成员，保留 CCPD 原始文件名（文件名本身编码了
  标注信息，见下节）。
- **是否入库**：这些图片**随仓库一起提供**（约 27 MB）。如需重新抽取，也可按下面的片段操作。

```python
import random

import remotezip

URL = "https://huggingface.co/datasets/okita-souji/ccpd2019train/resolve/main/train.zip"
with remotezip.RemoteZip(URL) as z:
    names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
    for name in random.sample(names, 300):
        base = name.rsplit("/", 1)[-1]
        with open(f"data/test_images/{base}", "wb") as f:
            f.write(z.read(name))
```

## 测试图片组织

> 仓库自带 `samples/demo2.jpg`，克隆后即可直接跑通方案一，**无需先准备本目录的图片**。

每位成员至少准备 5 张不同车辆图像，放入 `test_images/` 目录，并按下表分工命名，便于统计与追溯：

| 成员 | 命名示例 | 数量 |
| --- | --- | --- |
| 成员 A | `A_01.jpg` … `A_05.jpg` | 5 |
| 成员 B | `B_01.jpg` … `B_05.jpg` | 5 |
| 成员 C | `C_01.jpg` … `C_05.jpg` | 5 |
| 成员 D | `D_01.jpg` … `D_05.jpg` | 5 |

图片也可来自自行拍摄或视频截帧。`data/test_images/` 中随仓库提供的 300 张 CCPD 图片会进入版本管理；
成员自行拍摄或截帧的图片建议不要提交（体积较大，且可能涉及隐私）。

## 真值标注

CCPD 的文件名中编码了车牌字符与四角坐标，可使用 `scripts/parse_ccpd.py` 解析出每张图片的真实车牌号码，用于与识别结果比对、计算正确率。
