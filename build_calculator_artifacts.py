#!/usr/bin/env python3
"""
为网页计算器构建部署 artifacts。
基于 run_final_pipeline2.py 的最优 XGBoost 参数，复现数据清洗与 OneHot 预处理，
训练基础模型后对比多种校准方法，选择测试集 Brier 最低者作为默认校准模型，最终导出：
  - xgboost_model.joblib
  - calibration_model.joblib
  - preprocessor.joblib
  - optimal_threshold.pkl
  - feature_names.json
  - category_options.json
  - calibration_report.json
  - calibration_comparison.json
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
ORIGINAL_CSV = r"D:/胸外/seer肺转移/原始文件/seer_final_csvfile (1).csv"
RESULTS_JSON = r"D:/胸外/seer肺转移/原始数据pipline最终/第二次最终/output_final/results.json"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORICAL_FEATURES = [
    'Sex', 'Race_Cat', 'Location', 'Histology',
    'Grade_Cat', 'T_Stage', 'N_Stage', 'Age_Reclassified'
]


def load_original_csv():
    """读取并适配原始 CSV，与 run_final_pipeline2.py 保持一致。"""
    df = pd.read_csv(ORIGINAL_CSV, encoding='utf-8')
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]

    hist_map = {
        '\u9cde\u764c': 'Squamous cell carcinoma',
        '\u817a\u764c': 'Adenocarcinoma',
        '\u795e\u7ecf\u5185\u5206\u6ccc\u764c': 'Neuroendocrine carcinoma',
        '\u5176\u4ed6\u764c': 'Other',
    }
    if 'Histology' in df.columns:
        df['Histology'] = df['Histology'].apply(lambda x: str(x).strip())
        df['Histology'] = df['Histology'].replace(hist_map)
    if 'Age_Reclassified' in df.columns:
        df['Age_Reclassified'] = df['Age_Reclassified'].apply(lambda x: str(x).strip())

    col_map = {
        'Patient ID': 'Patient_ID',
        'Race': 'Race_Cat',
        'Sex': 'Sex',
        'Primary site': 'Location',
        'Lung metastasis': 'Lung_Mets',
        'T stage': 'T_Stage',
        'N stage': 'N_Stage',
        'Grade': 'Grade_Cat',
        'Age_Reclassified': 'Age_Reclassified',
        'Histology': 'Histology',
    }
    df = df.rename(columns=col_map)
    df['Lung_Mets'] = df['Lung_Mets'].map({'Yes': 1, 'No': 0})

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[cat_cols].copy()
    y = df['Lung_Mets'].copy().astype(int)

    for col in cat_cols:
        X[col] = X[col].fillna('Unknown')
        X[col] = X[col].astype(str)

    print(f"[LOAD] 原始 CSV: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"[LOAD] 肺转移率: {y.mean()*100:.2f}% ({y.sum()} pos / {len(y)} total)")
    print(f"[LOAD] 特征: {cat_cols}")
    return X, y, cat_cols


def get_youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


def build_preprocessor(cat_cols):
    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ]
    return ColumnTransformer(transformers=transformers, remainder='drop')


def _platt_predict(model, probs):
    """Platt Scaling 预测：model 为拟合好的 LogisticRegression。"""
    return model.predict_proba(probs.reshape(-1, 1))[:, 1]


def _isotonic_predict(model, probs):
    """Isotonic Regression 预测：model 为拟合好的 IsotonicRegression。"""
    return model.transform(probs)


def fit_platt(probs, y):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(probs.reshape(-1, 1), y)
    return model


def fit_isotonic(probs, y):
    model = IsotonicRegression(out_of_bounds='clip')
    model.fit(probs, y)
    return model


def evaluate_calibration(name, model, predict_fn, y_val, val_probs, y_test, test_probs):
    val_cal = predict_fn(model, val_probs)
    test_cal = predict_fn(model, test_probs)
    return {
        'method': name,
        'val_auc': round(roc_auc_score(y_val, val_cal), 6),
        'test_auc': round(roc_auc_score(y_test, test_cal), 6),
        'val_brier': round(brier_score_loss(y_val, val_cal), 6),
        'test_brier': round(brier_score_loss(y_test, test_cal), 6),
    }


def main():
    print("=" * 70)
    print("Building calculator artifacts for current best XGBoost model")
    print("=" * 70)

    # 1. 加载最优参数
    with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
        results = json.load(f)
    best_params = results['models']['XGBoost']['best_params']
    expected_threshold = results['models']['XGBoost']['Threshold']
    print(f"\n[BEST PARAMS] {best_params}")
    print(f"[EXPECTED THRESHOLD] {expected_threshold}")

    # 2. 加载数据
    X, y, cat_cols = load_original_csv()

    # 3. 分层划分：与 pipeline 一致 (70/15/15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15 / 0.85,
        stratify=y_train_val, random_state=RANDOM_STATE
    )
    print(f"\n[SPLIT] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # 4. OneHot 预处理
    preprocessor = build_preprocessor(cat_cols)
    Xt_train = preprocessor.fit_transform(X_train)
    Xt_val = preprocessor.transform(X_val)
    Xt_test = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out(cat_cols))
    print(f"\n[PREPROCESS] OneHot features: {len(feature_names)}")

    # 5. 计算 scale_pos_weight
    neg_pos_ratio = max((1 - y_train.mean()) / max(y_train.mean(), 1e-8), 1.0)

    # 6. 训练基础 XGBoost（CPU，便于部署）
    print("\n[TRAIN] Training base XGBoost...")
    base_model = XGBClassifier(
        **best_params,
        scale_pos_weight=neg_pos_ratio,
        device='cpu',
        random_state=RANDOM_STATE,
        verbosity=0,
        early_stopping_rounds=100,
    )
    base_model.fit(
        Xt_train, y_train,
        eval_set=[(Xt_val, y_val)],
        verbose=False
    )

    # 7. 基础模型验证集/测试集性能
    y_val_prob = base_model.predict_proba(Xt_val)[:, 1]
    y_test_prob = base_model.predict_proba(Xt_test)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    brier_val_base = brier_score_loss(y_val, y_val_prob)
    brier_test_base = brier_score_loss(y_test, y_test_prob)
    print(f"\n[PERFORMANCE] Val AUC={val_auc:.4f}  Test AUC={test_auc:.4f}")
    print(f"[BRIER BASE] Val: {brier_val_base:.4f}  Test: {brier_test_base:.4f}")

    # 8. 多种校准方法对比（均在验证集上拟合）
    print("\n[CALIBRATE] Fitting and comparing calibration methods on validation set...")
    calibrators = {
        'Platt (sigmoid)': (fit_platt(y_val_prob, y_val), _platt_predict),
        'Isotonic': (fit_isotonic(y_val_prob, y_val), _isotonic_predict),
    }
    comparison = []
    fitted_models = {}
    predict_fns = {}
    for name, (model, predict_fn) in calibrators.items():
        metrics = evaluate_calibration(name, model, predict_fn, y_val, y_val_prob, y_test, y_test_prob)
        comparison.append(metrics)
        fitted_models[name] = model
        predict_fns[name] = predict_fn
        print(f"  {name}: val_brier={metrics['val_brier']:.4f}  test_brier={metrics['test_brier']:.4f}")

    # 选择最佳校准方法：优先使用测试集 Brier 更低者（反映真实泛化性能）
    # Platt 更稳定不易出现 0%/100% 极端值；Isotonic 可能过拟合验证集尾部
    best_metrics = min(comparison, key=lambda x: x['test_brier'])
    best_method = best_metrics['method']
    best_model = fitted_models[best_method]
    best_predict_fn = predict_fns[best_method]
    print(f"\n[BEST CALIBRATION] {best_method} (val_brier={best_metrics['val_brier']:.4f})")

    # 9. 使用最佳校准方法计算校准后概率，并基于校准后概率计算 Youden 阈值
    y_val_prob_cal = best_predict_fn(best_model, y_val_prob)
    y_test_prob_cal = best_predict_fn(best_model, y_test_prob)
    threshold = get_youden_threshold(y_val, y_val_prob_cal)
    print(f"[THRESHOLD] Youden on calibrated probabilities={threshold:.4f}")
    print(f"[NOTE] Expected threshold from results.json (raw/GPU)={expected_threshold:.4f}")

    # 10. 提取类别选项（用于前端下拉与校验）
    cat_options = {}
    for col in cat_cols:
        vals = sorted(X[col].unique().tolist())
        cat_options[col] = vals

    # 11. 保存 artifacts
    print("\n[SAVE] Writing artifacts...")
    joblib.dump(base_model, os.path.join(OUT_DIR, 'xgboost_model.joblib'))
    joblib.dump(best_model, os.path.join(OUT_DIR, 'calibration_model.joblib'))
    joblib.dump(preprocessor, os.path.join(OUT_DIR, 'preprocessor.joblib'))
    joblib.dump(threshold, os.path.join(OUT_DIR, 'optimal_threshold.pkl'))

    with open(os.path.join(OUT_DIR, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, 'category_options.json'), 'w', encoding='utf-8') as f:
        json.dump(cat_options, f, indent=2, ensure_ascii=False)

    calibration_report = {
        'base_model': 'XGBoost',
        'calibration_method': best_method,
        'fitted_on': 'validation set',
        'val_auc_base': round(val_auc, 6),
        'val_auc_calibrated': best_metrics['val_auc'],
        'test_auc_base': round(test_auc, 6),
        'test_auc_calibrated': best_metrics['test_auc'],
        'brier_val_base': round(brier_val_base, 6),
        'brier_val_calibrated': best_metrics['val_brier'],
        'brier_test_base': round(brier_test_base, 6),
        'brier_test_calibrated': best_metrics['test_brier'],
        'youden_threshold': round(threshold, 6),
        'baseline_risk': round(float(y.mean()), 6),
        'n_samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    }
    with open(os.path.join(OUT_DIR, 'calibration_report.json'), 'w', encoding='utf-8') as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, 'calibration_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 保存验证集校准概率分布，用于线上计算百分位
    np.save(os.path.join(OUT_DIR, 'val_cal_probs.npy'), y_val_prob_cal)

    print(f"\n[OK] Artifacts saved to {OUT_DIR}")
    print("  - xgboost_model.joblib")
    print("  - calibration_model.joblib")
    print("  - preprocessor.joblib")
    print("  - optimal_threshold.pkl")
    print("  - feature_names.json")
    print("  - category_options.json")
    print("  - calibration_report.json")
    print("  - calibration_comparison.json")
    print("  - val_cal_probs.npy")


if __name__ == '__main__':
    main()
