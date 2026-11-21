# 队友任务说明（可直接转发）

> 本文包含三份独立任务，每节可单独复制转发。所有命令均在**仓库根目录**、**已激活项目 conda 环境**后运行。
> 通用前置：先 `git pull` 拉取最新代码，确认当前在 `license-plate-recognition` 根目录。

---

## 任务一：三组案例标注图

三个案例已选定，逐个跑出标注图即可。**不要加 `--no-save`，不要加 `--tile`，单文件运行两个脚本各一遍。**

### 案例 1 · 成功案例（停车场多车牌，整图模式两方案都全对）

```bash
python scripts/run_two_stage.py --source data/test_images/demo2.jpg
python scripts/run_integrated.py --source data/test_images/demo2.jpg
```

### 案例 2 · 倾斜失败

真值 `皖AS821X`，倾斜 23°：方案二全错、方案一正确。

```bash
python scripts/run_two_stage.py --source "data/test_images/0304238505747-94_67-134&463_515&555-542&568_141&534_90&445_491&479-0_0_16_32_26_25_21-128-157.jpg"
python scripts/run_integrated.py --source "data/test_images/0304238505747-94_67-134&463_515&555-542&568_141&534_90&445_491&479-0_0_16_32_26_25_21-128-157.jpg"
```

### 案例 3 · 省份字失败

真值 `川J6MS71`，完全水平：两方案都错成 `浙J6MS71`。

```bash
python scripts/run_two_stage.py --source "data/test_images/00946479885057-90_90-183&574_354&644-362&638_187&646_188&580_363&572-11_8_30_11_16_31_25-163-9.jpg"
python scripts/run_integrated.py --source "data/test_images/00946479885057-90_90-183&574_354&644-362&638_187&646_188&580_363&572-11_8_30_11_16_31_25-163-9.jpg"
```

### 交付物

- **6 张标注图**（3 案例 × 2 方案），改成看得懂的名字（如 `case2_tilt_two_stage.jpg`），放进一个文件夹。
- 附一行说明，记录每个案例的**真值**与**两方案输出**。
- 用途：案例 2、3 的图是成员 D 写失败案例分析的素材；案例 1 的图给报告第三节当流程演示。

---

## 任务二：汇总四人亲测记录

1. 先定统一记录格式，每人按此提交：**图片文件名 / 真实号码 / 场景 / 方案一结果 / 方案二结果**（对错由汇总人在合并时判定，整牌全对才算对）。
2. 每位成员自拍 **≥5 张**不同车辆，单独文件夹 + `labels.csv` 记真值，两个方案各跑一遍，把结果 CSV 和记录交给汇总人。
3. 汇总人合并成报告第四节的统计表：**每人一行**（测试数、方案一正确数、方案二正确数、两个正确率），最后一行合计。
4. 抽查每人至少 1 张：照片和所填真值是否对得上，防止号码记错污染统计。

---

## 任务三：拓展实验（OpenCV 增强 · 偏暗子集）

先 `git pull` 拿到 `enhance_images.py`，然后跑 A、B 两档。基线是主报告里方案二偏暗的 **91.9%（111 张），不用重跑**。

### A、B 两档增强

```bash
python scripts/enhance_images.py --source data/test_images --output data/enhanced_dark_a --brightness-max 100 --clahe
python scripts/enhance_images.py --source data/test_images --output data/enhanced_dark_b --brightness-max 100 --clahe --gamma 0.6
```

### 两档识别（只出 CSV，不写结果图）

```bash
python scripts/run_two_stage.py --source data/enhanced_dark_a --no-save --csv samples/results/two_stage_dark_a.csv
python scripts/run_two_stage.py --source data/enhanced_dark_b --no-save --csv samples/results/two_stage_dark_b.csv
```

### 生成评估报告

```bash
python scripts/evaluate.py --results samples/results/two_stage_dark_a.csv --output samples/results/eval_dark_a.md
python scripts/evaluate.py --results samples/results/two_stage_dark_b.csv --output samples/results/eval_dark_b.md
```

### 交付物

- 两份 `eval_dark` 报告提交进仓库。
- 一张**三行对比表**：基线 91.9% / A / B 的整牌正确率与字符准确率。
- **两组对照图**：一张被增强救回来的、一张没救的（原图 + 同名增强图 + 标注结果图）。
- **§四.4 分析文字**：涨了分析为什么；没涨也照实写（如 CLAHE 放大噪声），负结果加分析同样成立。

---

## 衔接关系提醒

成员 D 写第四节失败案例分析要用成员 C 的案例图，成员 C 的统计表要等四个人都交记录——这两人是链条的下游。自己的自拍记录（已完成 2/5）先交齐，别让成员 C 等。
