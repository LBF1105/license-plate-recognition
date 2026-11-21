# 增强前后对照图

每组三张：`original.jpg`（原图）、`enhanced_b.jpg`（B 档 Gamma 0.6 + CLAHE 增强图）、`result_b.jpg`（增强图上的识别标注结果）。

## rescued/（被增强救回）

- 真值：`皖AVS180`
- 基线（未增强）：`赣AVS180`，省份字错误。
- B 档增强后：`皖AVS180`，正确。

## failed/（增强也没救回）

- 真值：`皖AFU995`
- 基线（未增强）：`豫AFU995`。
- B 档增强后：`赣AFU995`，省份字仍错误，属 LPRNet 字符分类层面的混淆。

> 完整分析见 `docs/enhance_experiment.md`。
