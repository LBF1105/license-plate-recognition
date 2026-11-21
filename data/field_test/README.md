# 四人亲测记录（任务二）

每位成员至少自拍 **5 张**不同车辆的照片，两个方案各跑一遍，由汇总人统一计算正确率。

## 目录结构

```
data/field_test/
├── member_A/
│   ├── images/                  # 放自己拍的照片
│   ├── labels.csv               # 手工填写真值（文件名、真实号码、场景）
│   ├── result_integrated.csv    # 方案一结果（脚本生成）
│   └── result_two_stage.csv     # 方案二结果（脚本生成）
├── member_B/ ...
├── member_C/ ...
└── member_D/ ...
```

> 成员目录可按实际姓名重命名（如 `zhangsan`），重命名后命令里的路径同步修改即可。

## 每人操作步骤（以 member_A 为例）

1. 把自拍照片（≥5 张）放进 `data/field_test/member_A/images/`。
2. 打开 `data/field_test/member_A/labels.csv`，逐张填写：
   - `filename`：图片文件名（含扩展名，与 images 里完全一致）
   - `true_plate`：真实车牌号码（整牌，含省份汉字）
   - `scene`：场景描述，如 正面水平 / 侧面倾斜 / 偏暗 / 远距离
3. 两个方案各跑一遍，结果 CSV 输出到自己的目录：

```bash
python scripts/run_integrated.py --source data/field_test/member_A/images --no-save --csv data/field_test/member_A/result_integrated.csv
python scripts/run_two_stage.py --source data/field_test/member_A/images --no-save --csv data/field_test/member_A/result_two_stage.csv
```

## 汇总（汇总人执行）

四人都交齐后，运行统计脚本自动生成每人一行的统计表（含合计）：

```bash
python scripts/aggregate_field_test.py --field-root data/field_test --output samples/results/field_test_summary.md
```

## 判定与抽查

- 整牌字符全部一致才算正确，错一个字符即算错。
- 汇总人抽查每人至少 1 张：照片与 `labels.csv` 所填真值是否对得上，防止号码记错污染统计。
