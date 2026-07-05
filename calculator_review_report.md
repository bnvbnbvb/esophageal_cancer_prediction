# 网页计算器复核报告

**复核对象**：`D:/胸外/seer肺转移/原始数据pipline最终/第二次最终/web_calculator/`

**复核日期**：2026-07-01

**复核结论**：**已修复关键问题，当前版本可本地运行，建议部署前再做一次完整回归测试。**

---

## 1. 后端逻辑复核

### 1.1 模型与论文一致性
| 检查项 | 状态 | 说明 |
|---|---|---|
| 模型类型 | ✅ | XGBoost，与 `run_final_pipeline2.py` 最优模型一致 |
| 特征集合 | ✅ | 8 个分类变量：Sex、Race_Cat、Location、Histology、Grade_Cat、T_Stage、N_Stage、Age_Reclassified |
| 训练参数 | ✅ | 来自 `output_final/results.json` 中 XGBoost 的 `best_params` |
| 预处理 | ✅ | `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`，与论文一致 |
| 数据清洗 | ✅ | 列名映射、Histology 中文→英文映射、分层划分均与 `run_final_pipeline2.py` 一致 |
| 重训练差异 | ⚠️ | 使用 CPU 而非原 GPU，`device='cpu'` 导致概率分布略有偏移；AUC 基本一致（Test 0.7529 vs 论文 0.7535） |

### 1.2 概率校准
| 检查项 | 状态 | 说明 |
|---|---|---|
| 校准方法 | ✅ | Platt Scaling（单变量 LogisticRegression） |
| 校准拟合集 | ✅ | 验证集（val），未 leakage 到测试集 |
| 校准效果 | ✅ | Brier score（test）从 0.2041 降至 0.0565 |
| AUC 影响 | ✅ | 校准前后 AUC 不变，仅调整概率刻度 |

### 1.3 阈值与决策逻辑（关键修复）
| 检查项 | 状态 | 说明 |
|---|---|---|
| 阈值来源 | ✅ 已修复 | 最初错误地使用了未校准概率的 Youden 阈值（0.5032），导致所有校准后概率都低于阈值、全部预测为阴性。已修复为基于**校准后概率**重新计算 Youden 阈值（0.0668）。 |
| 决策一致性 | ✅ 已修复 | 现在分类决策与展示的概率使用同一刻度，逻辑自洽。 |
| 性能指标 | ✅ | 新阈值下 Test AUC=0.7529，Acc=0.7562，Sens=0.6724，Spec=0.7618，PPV=0.1605，NPV=0.9717 |

### 1.4 输入校验与错误处理
| 检查项 | 状态 | 说明 |
|---|---|---|
| 缺失字段 | ✅ 已修复 | 最初返回 HTTP 500；已改为返回 HTTP 400，并提示缺失字段 |
| 无效类别值 | ✅ 已修复 | 最初返回 HTTP 500；已改为返回 HTTP 400，并提示可选值 |
| 批量预测异常隔离 | ✅ | `batch_predict` 单条异常不影响其他条目 |
| 字段映射 | ✅ | `FIELD_MAPPING` 覆盖前端字段名与模型特征名 |
| 类别值映射 | ✅ 已清理 | `Histology` 大小写/“Other Carcinoma”→“Other” 映射正确；移除了冗余的 `Location` 映射 |

---

## 2. 前端信息与结果判读复核

### 2.1 信息准确性
| 检查项 | 状态 | 说明 |
|---|---|---|
| 模型声明 | ✅ | 已改为 XGBoost / Stratified Split / Youden Threshold / Platt-Calibrated |
| 样本量/AUC | ✅ | 12,187 Patients / Val AUC 0.76 |
| SMOTE 残留 | ✅ | 已移除 |
| 免责声明 | ✅ 已更新 | 已说明“未外部验证”“概率为群体估计”“最高概率约 20%” |

### 2.2 下拉选项与后端对齐
| 检查项 | 状态 | 说明 |
|---|---|---|
| Race / Sex / T Stage / N Stage / Grade / Age | ✅ | 与模型类别完全一致 |
| Location | ✅ | HTML value 与模型值一致（"Abdominal/overlapping esophagus"） |
| Histology | ✅ | HTML 显示文本大写，通过 `VALUE_NORMALIZE` 映射到模型值 |

### 2.3 结果判读（本次重点）

#### 问题发现
校准后概率分布发生显著变化：
- 中位数：~5.9%
- P90：~13.6%
- P95：~16.3%
- 最大值：~22.4%

如果沿用原 <20% / 20-50% / ≥50% 的风险分层，**几乎所有结果都是“低风险”**，无法起到风险分层作用。

#### 已修复
已将风险分层调整为与校准后分布匹配的三档：
- **低风险**：< 5%
- **中风险**：5% - 15%
- **高风险**：≥ 15%

#### 临床建议已同步调整
- 避免“高风险=立即 PET-CT/MDT”等过度强烈的表述
- 强调“最高预测概率约 20%”“需结合临床综合判断”

---

## 3. 部署配置复核

| 文件 | 状态 | 说明 |
|---|---|---|
| `requirements.txt` | ✅ | 含 `xgboost==3.2.0`、`scikit-learn==1.9.0`，已移除 `imbalanced-learn` |
| `Procfile` | ✅ 已修复 | 原 `web gunicorn appapp` 已改为 `web: gunicorn app:app` |
| `render.yaml` | ✅ | `gunicorn app:app` 正确，Python 3.10.0 |
| `runtime.txt` | ⚠️ | `python-3.10.0` 较旧；`requirements.txt` 指定了较新包版本，建议 Render 上选择 Python 3.10.12 或确认 3.10.0 能安装 sklearn 1.9 / xgboost 3.2.0 |
| 模型 artifacts | ✅ | `xgboost_model.joblib`、`platt_model.joblib`、`preprocessor.joblib`、`optimal_threshold.pkl`、JSON 配置齐全 |

---

## 4. 功能测试结果

| 接口/场景 | 状态 | 结果 |
|---|---|---|
| `GET /health` | ✅ | 返回 healthy，阈值 0.0668 |
| `GET /model_info` | ✅ | 返回模型信息、校准状态、性能指标 |
| `POST /predict` 低风险 | ✅ | 概率 2.43%，prediction=0，低风险 |
| `POST /predict` 中风险 | ✅ | 概率 12.93%，prediction=1，中风险 |
| `POST /predict` 高风险 | ✅ | 概率 22.4%，prediction=1，高风险 |
| `POST /predict` 缺失字段 | ✅ | HTTP 400，提示缺失字段 |
| `POST /predict` 无效类别 | ✅ | HTTP 400，提示可选值 |
| `POST /batch_predict` | ✅ | 逐条返回，异常条目不影响其他 |
| 所有类别组合 | ✅ | 5×4×4×4×5×4×5 = 6400 种组合映射均通过 |

---

## 5. 后续优化建议（按优先级）

### 高优先级
1. **外部验证**
   - 当前模型仅在 SEER 内部训练/验证/测试，尚未经过外部独立队列验证。
   - 建议：在页面前端和论文中明确标注“未外部验证”，并尽快收集独立中心数据验证。

2. **阈值临床适用性再评估**
   - 当前 Youden 阈值 6.68% 接近患病率 6.32%，约一半患者会被判为阳性。
   - 建议：根据临床使用场景（筛查 vs 确诊后分层）选择不同阈值，或提供“敏感性优先/特异性优先”切换开关。

### 中优先级
3. **校准方法补充对比**
   - 当前仅使用 Platt Scaling。
   - 建议：在验证集上对比 Isotonic Regression、Beta calibration，选择 Brier 最低的方法。

4. **结果展示增强**
   - 当前仅展示绝对概率。
   - 建议：增加“与同队列相比的相对风险/百分位”或“近似 1 年内风险”解释，帮助用户理解 22% 的含义。

5. **SHAP 解释**
   - 当前结果无特征贡献解释。
   - 建议：增加 SHAP 值展示，说明哪些特征推高了/拉低了该患者的风险。

### 低优先级
6. **前端图表**
   - 可在结果页增加校准曲线、ROC 曲线链接，提升专业感。

7. **多语言支持**
   - 当前界面为英文，临床用户可能更习惯中文；可考虑中英切换。

---

## 6. 最终结论

- **已修复的关键问题**：
  1. `optimal_threshold.pkl` 在清理旧文件时被误删，已重新生成。
  2. 阈值计算从未校准概率迁移到校准后概率，避免所有预测均为阴性。
  3. 风险分层从 <20%/20-50%/≥50% 调整为 <5%/5-15%/≥15%，与校准后分布匹配。
  4. 输入校验错误从 HTTP 500 改为 HTTP 400。
  5. 临床建议已根据校准后概率幅度调整，避免过度解读。

- **当前状态**：本地 Flask 服务运行正常，接口测试通过，可直接在浏览器中访问 `http://localhost:5000`。

- **部署建议**：
  - 在 Render 上先创建测试服务，验证依赖安装无报错。
  - 部署后访问 `/model_info` 和 `/health` 确认模型加载成功。
  - 如需展示最新修复，请重新上传 `web_calculator/` 目录到 GitHub 并触发 Render 重新部署。
