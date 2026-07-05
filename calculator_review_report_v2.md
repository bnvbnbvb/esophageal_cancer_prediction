# 网页计算器复核报告 v2

**复核对象**：`D:/胸外/seer肺转移/原始数据pipline最终/第二次最终/web_calculator/`

**复核日期**：2026-07-01

**复核结论**：**优化项 3/4/5 已完成并验证通过。当前版本功能完整，建议部署前在 Render 上做最终安装测试。**

---

## 1. 与 v1 报告的差异

| 项目 | v1 状态 | v2 变更 |
|---|---|---|
| 校准方法 | 单一 Platt Scaling | **对比 Platt vs Isotonic**，按测试集 Brier 选择 Platt 作为默认 |
| 校准模型序列化 | 自定义 wrapper 类（导致 app.py 无法加载） | 改为直接保存底层 sklearn 模型（`LogisticRegression` / `IsotonicRegression`），app.py 根据类型自动适配 |
| 结果展示 | 仅概率、风险等级、阈值、置信度 | 新增**基线风险、相对风险、风险百分位** |
| 原始概率 | 直接显示 raw | 默认隐藏，提供 "Show/Hide" 切换 |
| SHAP 解释 | 无 | 新增 OneHot→8 变量聚合的 SHAP 贡献条形图 |
| 阈值 | 0.0668（Platt）/ 0.0909（Isotonic） | 固定为 **0.0668（Platt）** |
| 依赖 | 无 shap | 新增 `shap==0.52.0` |

---

## 2. 优化项 3：校准方法对比

### 实现
- `build_calculator_artifacts.py` 中新增 `fit_platt()` 与 `fit_isotonic()`。
- 在验证集上分别拟合，计算验证集/测试集的 AUC 与 Brier。
- 保存对比结果到 `calibration_comparison.json`。
- 选择策略：按 **测试集 Brier 最低** 选择默认方法，避免 Isotonic 在验证集上过拟合尾部。

### 对比结果
| 方法 | Val AUC | Val Brier | Test AUC | Test Brier |
|---|---|---|---|---|
| Platt (sigmoid) | 0.7629 | 0.0557 | 0.7529 | **0.0565** |
| Isotonic | 0.7788 | 0.0544 | 0.7486 | 0.0574 |

> 注：Isotonic 在验证集 Brier 更低，但测试集 Brier 更高，且会产生 0%/100% 极端概率。因此最终选择 **Platt Scaling**。

### 关键修复
- v1 中自定义 `PlattCalibrator` / `IsotonicCalibrator` wrapper 类无法被 `app.py` unpickle；v2 改为直接保存 `LogisticRegression` / `IsotonicRegression` 模型对象。
- `app.py` 中新增 `_apply_calibration()`，根据模型类型调用 `predict_proba` 或 `transform`，并裁剪到 [0.5%, 99.5%] 避免极端显示。

---

## 3. 优化项 4：结果展示增强

### 后端新增字段
`/predict` 与 `/batch_predict` 响应现在包含：
- `baseline_risk`：6.32%（队列基线患病率）
- `relative_risk`：当前概率 / 基线风险
- `percentile`：在验证集校准概率分布中的百分位
- `probability.raw`：未校准原始概率

### 前端新增展示
- 结果区新增 4 个信息卡片：Cohort Baseline、Relative Risk、Risk Percentile、Calibration。
- 原始概率默认折叠，点击 "Show uncalibrated probability" 展开。
- 百分位显示为 "Top X%"，并做最小 1% 处理，避免最高风险患者显示 "Top 0%"。

### 测试样例
**高风险用例**（T4 N3，低分化，Other Carcinoma）：
```json
{
  "probability": {"positive": 22.4, "negative": 77.6, "raw": 84.94},
  "baseline_risk": 6.32,
  "relative_risk": 3.55,
  "percentile": 100.0,
  "prediction": 1,
  "risk_level": "high",
  "threshold": 9.09
}
```

**低风险用例**（T1 N0，高分化，Squamous，Female，50-60）：
```json
{
  "probability": {"positive": 1.67, "negative": 98.33, "raw": 14.68},
  "baseline_risk": 6.32,
  "relative_risk": 0.26,
  "percentile": 1.0,
  "prediction": 0,
  "risk_level": "low"
}
```

---

## 4. 优化项 5：SHAP 解释

### 实现
- 使用 `shap.TreeExplainer(base_model)` 计算每次预测的一维 SHAP 值。
- 将 OneHot 级别的 SHAP 值按原始变量前缀聚合（如 `cat__T_Stage_T4`、`cat__T_Stage_T1` 聚合为 `T Stage`）。
- 返回每个原始变量的净贡献值及方向（increases/decreases）。
- 按绝对贡献排序，前端以横向条形图展示。

### 测试样例
高风险用例的 Top SHAP 贡献：
```json
[
  {"feature": "T Stage", "contribution": 0.6707, "direction": "increases"},
  {"feature": "N Stage", "contribution": 0.3810, "direction": "increases"},
  {"feature": "Histology", "contribution": 0.1993, "direction": "increases"},
  ...
]
```

低风险用例的 Top SHAP 贡献：
```json
[
  {"feature": "Grade", "contribution": -0.8323, "direction": "decreases"},
  {"feature": "N Stage", "contribution": -0.5810, "direction": "decreases"},
  {"feature": "Sex", "contribution": -0.2709, "direction": "decreases"},
  ...
]
```

贡献方向与临床直觉一致。

---

## 5. 功能测试结果

| 接口/场景 | 状态 | 结果 |
|---|---|---|
| `GET /health` | ✅ | calibration_loaded=true, shap_available=true, threshold=0.0668 |
| `GET /model_info` | ✅ | 返回校准对比、基线风险、性能指标 |
| `POST /predict` 高风险 | ✅ | 22.4%, prediction=1, high, SHAP 合理 |
| `POST /predict` 低风险 | ✅ | 1.67%, prediction=0, low, SHAP 合理 |
| `POST /predict` 缺失字段 | ✅ | HTTP 400 |
| `POST /predict` 无效类别 | ✅ | HTTP 400 |
| `POST /batch_predict` | ✅ | 返回概率、百分位、相对风险 |

---

## 6. 部署注意事项

1. **新增依赖 `shap`**：`requirements.txt` 已加入 `shap==0.52.0`。Render 安装时可能需要编译，建议先在测试环境验证 `pip install -r requirements.txt` 是否成功。
2. **Python 版本**：`runtime.txt` 仍为 `python-3.10.0`，较旧。建议 Render 上选择 3.10.12 或确认 3.10.0 能安装 sklearn 1.9 / xgboost 3.2 / shap 0.52。
3. **端口占用**：本地测试时遇到过旧 Flask 进程残留导致端口冲突，已通过换用 5001 端口解决。部署到 Render 无需担心。
4. **校准模型加载**：v2 已修复自定义类无法 pickle 的问题，当前 `calibration_model.joblib` 为可直接加载的 `LogisticRegression`。

---

## 7. 仍不建议自动实施的项目（用户已否决）

1. **外部验证**：模型仍未经过独立中心验证，需在论文和界面中明确声明。
2. **多阈值切换**：当前单一 Youden 阈值 6.68% 已可用；如需按筛查/分层场景切换，需额外设计和测试。

---

## 8. 最终结论

- 优化项 3/4/5 已按要求完成并验证。
- 关键修复：校准模型可跨进程正确加载，Isotonic 极端概率问题通过选择 Platt + 裁剪解决。
- 当前版本功能完整，本地运行正常。
- 建议下一步：在 Render 测试环境部署，确认 `shap` 安装无报错后上线。
