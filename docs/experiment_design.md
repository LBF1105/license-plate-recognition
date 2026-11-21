# 对照实验设计

## 一、实验目的

在同一组测试图片、同一运行设备下，比较两种车牌识别方案的整牌识别正确率、单图平均耗时与失败案例，理解集成式框架与分步式架构的差异。

## 二、两个方案

| 方案 | 架构 | 特征 |
| --- | --- | --- |
| 方案一（基线） | HyperLPR3 集成式 | 框架内部端到端完成检测与识别，调用简单 |
| 方案二（对照） | YOLO 检测 + LPRNet 识别 | 检测与识别拆分为独立模块，可分别替换、调优 |

## 三、控制变量

为保证对照公平，两方案必须满足：

1. 使用**完全相同**的测试图片集
2. 在同一台设备、同一运行模式（CPU/GPU）下运行
3. 使用统一的可视化与评估口径（common 模块）
4. 正确判定标准一致：整牌字符全部正确才计为正确

## 四、评估指标

- **整牌识别正确率**：`正确图片数 ÷ 测试图片总数 × 100%`
- **单图平均耗时**：从读入到输出结果的时间，逐张计时后取平均
- **失败案例**：记录漏检、误识的图片及原因（模糊、倾斜、光照、遮挡等）

## 五、结果记录模板

| 方案 | 测试数量 | 正确数量 | 正确率 | 平均耗时（ms/张） |
| --- | --- | --- | --- | --- |
| 集成式 HyperLPR3 | 300 | 289 | 96.3% | 80 |
| 分步式 YOLOv8+LPRNet | 300 | 273 | 91.0% | 60 |

（2026-11 实测，测试集为 CCPD 模糊子集 300 张；完整报告见 `samples/results/`
下的 `eval_integrated.md`、`eval_two_stage.md`、`comparison.md`。）

## 六、执行方式

```bash
# 1. 两方案分别批量识别（只输出 CSV）
python scripts/run_integrated.py --source data/test_images --no-save --csv samples/results/integrated_ccpd.csv
python scripts/run_two_stage.py --source data/test_images --no-save --csv samples/results/two_stage_ccpd.csv
# 2. 各自对照 CCPD 真值评估（真值编码在文件名中，由 evaluate.py 内部解析）
python scripts/evaluate.py --results samples/results/integrated_ccpd.csv --output samples/results/eval_integrated.md
python scripts/evaluate.py --results samples/results/two_stage_ccpd.csv --output samples/results/eval_two_stage.md
# 3. 生成并排对照报告（总览 + 分场景 + 失败交集）
python scripts/compare.py --integrated samples/results/integrated_ccpd.csv --two-stage samples/results/two_stage_ccpd.csv --output samples/results/comparison.md
```

对照结果输出到 `samples/results/`。若某方案增强后没有提升，分析其原因同样属于有效实验结论。
