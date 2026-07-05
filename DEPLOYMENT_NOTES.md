# 食管癌肺转移风险预测网页计算器 - 更新说明

## 本次更新内容
1. **模型替换**：由旧版 Random Forest 替换为当前论文定稿的 XGBoost 模型。
2. **预处理升级**：由手写 label encoding 改为与论文一致的 `OneHotEncoder`。
3. **概率校准对比**：在验证集上对比 Platt Scaling 与 Isotonic Regression，按测试集 Brier 最低选择默认校准方法（当前为 Platt Scaling）。
4. **结果展示增强**：新增队列基线风险、相对风险倍数、风险百分位。
5. **SHAP 解释**：每次预测返回 8 个原始变量的特征贡献，前端以条形图展示。
6. **前端信息更新**：样本量、AUC、模型类型、风险分层、免责声明等均已同步。

## 模型性能（CPU 重训练复现）
- 数据来源：SEER 数据库 12,187 例食管癌患者
- 肺转移率：6.32%（770 例阳性）
- 验证集 AUC：0.7629
- 测试集 AUC：0.7529
- 默认校准方法：Platt Scaling（val_brier=0.0557，test_brier=0.0565）
- Isotonic Regression 对比：val_brier=0.0544，test_brier=0.0574
- Youden 阈值（基于 Platt 校准后概率）：0.0668（6.68%）
- 风险分层（基于校准后概率）：低风险 <5%，中风险 5-15%，高风险 ≥15%

> 注：原始 `run_final_pipeline2.py` 在 GPU 环境下、基于未校准概率得到的 Youden 阈值为 0.5383；本次为 Render CPU 部署重新训练，并基于**校准后概率**重新计算 Youden 阈值，AUC 基本一致。这样可保证分类决策与展示的概率刻度一致。

## 文件清单
| 文件 | 说明 |
|------|------|
| `app.py` | Flask 后端主程序（含校准、SHAP、风险解释） |
| `templates/index.html` | 前端页面 |
| `requirements.txt` | Python 依赖 |
| `Procfile` | Render/Gunicorn 启动命令 |
| `render.yaml` | Render 部署配置 |
| `runtime.txt` | Python 运行时版本 |
| `xgboost_model.joblib` | XGBoost 基础模型 |
| `calibration_model.joblib` | 选定的校准模型（当前为 Platt Scaling） |
| `preprocessor.joblib` | OneHot 预处理器 |
| `optimal_threshold.pkl` | Youden 阈值 |
| `val_cal_probs.npy` | 验证集校准概率分布（用于计算百分位） |
| `category_options.json` | 前端下拉选项与模型类别对照 |
| `feature_names.json` | OneHot 展开特征名 |
| `calibration_report.json` | 校准性能报告 |
| `calibration_comparison.json` | 各校准方法对比结果 |
| `build_calculator_artifacts.py` | 复现训练与导出 artifacts 的脚本 |

## 本地运行
```bash
pip install -r requirements.txt
python app.py
```
访问 http://localhost:5000

## Render 部署
1. 将本目录推送到 GitHub。
2. 在 Render 创建 Python Web Service，指向本目录。
3. 构建命令：`pip install -r requirements.txt`
4. 启动命令：`gunicorn app:app`
5. 环境变量（可选）：
   - `CALIBRATE=0`：关闭概率校准，返回原始 XGBoost 概率。
   - `CALIBRATE=1`（默认）：返回选定校准方法的概率。

## 重要说明
- 当前模型**未经过外部独立队列验证**。
- 预测结果仅供临床参考，不能替代专业医疗诊断。
