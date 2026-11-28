# 自拍测试集（湘牌）

组长（成员 A）亲测用的自拍图片，与 CCPD 测试集分开报告。
图片为真实拍摄（2026-11-21 两张、2026-11-28 六张），不套用 CCPD 文件名格式，
真值见 `labels.csv`。

## 图片清单

| 文件 | 真值 | 场景 | 类型 |
| --- | --- | --- | --- |
| xiang_01.png | 湘A7129N | 白天正常 | 蓝牌单层 |
| xiang_02.png | 湘C782J1 | 白天正常 | 蓝牌单层 |
| xiang_03.jpg | 湘CF86086 | 白天正常 | 绿牌单层 |
| xiang_04.jpg | 湘C53K67 | 斜角度 | 蓝牌单层 |
| xiang_05.jpg | 湘BH8296 | 斜角度 | 蓝牌单层 |
| xiang_06.jpg | 湘ATL735 | 远距离 | 蓝牌单层 |
| xiang_07.jpg | 湘CA331L | 半遮挡 | 蓝牌单层 |
| xiang_08.jpg | 湘CC897V | 白天正常 | 蓝牌单层 |

## 真值待核对

`labels.csv` 里的号码是读图整理出来的，请队友（尤其成员 C）逐张对照照片核对，
确认无误后再用于报告和统计。如有读错，直接改 `labels.csv` 后重新打分。

## 怎么跑、怎么打分

```bash
# 两个方案各跑一遍（方案一不用 --tile，与正式对照保持一致）
python scripts/run_integrated.py --source data/test_images_xiang --csv samples/results/integrated_xiang.csv
python scripts/run_two_stage.py --source data/test_images_xiang --csv samples/results/two_stage_xiang.csv
# 打分
python scripts/score_xiang.py
```

## 注意

- xiang_04、xiang_08 的画面里除了目标车牌还有背景车牌，
  `score_xiang.py` 按图分组计分：一张图只要有任意一个检出等于真值就算对，
  其余检出会在表里列出来，不会悄悄丢掉。
- 未入选的备用照片：双绿牌同框那张（一图两牌，打分脚本按一图一真值设计，
  暂不进本集）、远处蔚来绿牌那张（车牌太小，读数不可靠）。
