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

## 测试图片组织

> 仓库自带 `samples/demo2.jpg`，克隆后即可直接跑通方案一，**无需先准备本目录的图片**。

每位成员至少准备 5 张不同车辆图像，放入 `test_images/` 目录，并按下表分工命名，便于统计与追溯：

| 成员 | 命名示例 | 数量 |
| --- | --- | --- |
| 成员 A | `A_01.jpg` … `A_05.jpg` | 5 |
| 成员 B | `B_01.jpg` … `B_05.jpg` | 5 |
| 成员 C | `C_01.jpg` … `C_05.jpg` | 5 |
| 成员 D | `D_01.jpg` … `D_05.jpg` | 5 |

图片也可来自自行拍摄或视频截帧。`test_images/` 目录中的图片不纳入 Git，仅保留 `.gitkeep` 占位文件。

## 真值标注

CCPD 的文件名中编码了车牌字符与四角坐标，可使用 `scripts/parse_ccpd.py` 解析出每张图片的真实车牌号码，用于与识别结果比对、计算正确率。
