from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os
import warnings
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


def _apply_calibration(raw_prob):
    """根据 calibration_model 类型应用校准，并裁剪到合理范围避免 0%/100% 极端显示。"""
    if calibration_model is None:
        return float(raw_prob)
    raw_prob = float(raw_prob)
    if isinstance(calibration_model, LogisticRegression):
        prob = float(calibration_model.predict_proba(np.array([[raw_prob]]))[0, 1])
    elif isinstance(calibration_model, IsotonicRegression):
        prob = float(calibration_model.transform(np.array([raw_prob]))[0])
    else:
        # 兜底：尝试调用 transform 或 predict_proba
        if hasattr(calibration_model, 'transform'):
            prob = float(calibration_model.transform(np.array([raw_prob]))[0])
        elif hasattr(calibration_model, 'predict_proba'):
            prob = float(calibration_model.predict_proba(np.array([[raw_prob]]))[0, 1])
        else:
            prob = float(raw_prob)
    # 裁剪到 [0.5%, 99.5%]，避免前端显示 0% 或 100%
    return float(np.clip(prob, 0.005, 0.995))


# ============ 辅助函数：转换 NumPy 类型 ============
def convert_to_python_type(obj):
    """将 NumPy 类型转换为 Python 原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(i) for i in obj]
    else:
        return obj


# ============ 加载模型和配置文件 ============
print("正在加载模型文件...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_joblib(name, default=None):
    path = os.path.join(BASE_DIR, name)
    try:
        obj = joblib.load(path)
        print(f"[OK] {name} loaded")
        return obj
    except Exception as e:
        print(f"[FAIL] {name} load failed: {e}")
        return default


def _load_json(name, default=None):
    path = os.path.join(BASE_DIR, name)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        print(f"[OK] {name} loaded")
        return obj
    except Exception as e:
        print(f"[FAIL] {name} load failed: {e}")
        return default


base_model = _load_joblib('xgboost_model.joblib')
calibration_model = _load_joblib('calibration_model.joblib')
preprocessor = _load_joblib('preprocessor.joblib')
optimal_threshold = _load_joblib('optimal_threshold.pkl', 0.5)
optimal_threshold = float(optimal_threshold)
feature_names = _load_json('feature_names.json', [])
category_options = _load_json('category_options.json', {})
calibration_report = _load_json('calibration_report.json', {})
calibration_comparison = _load_json('calibration_comparison.json', [])

# 加载验证集校准概率分布，用于计算百分位
val_cal_probs_path = os.path.join(BASE_DIR, 'val_cal_probs.npy')
val_cal_probs = None
try:
    val_cal_probs = np.load(val_cal_probs_path)
    print("[OK] val_cal_probs.npy loaded")
except Exception as e:
    print(f"[FAIL] val_cal_probs.npy load failed: {e}")

# 队列基线风险
BASELINE_RISK = calibration_report.get('baseline_risk', 0.0632)

# 是否启用校准：默认开启，可通过环境变量 CALIBRATE=0 关闭
USE_CALIBRATION = os.environ.get('CALIBRATE', '1') == '1'

# 可选：加载 SHAP
HAS_SHAP = False
shap = None
try:
    import shap
    HAS_SHAP = True
    print("[OK] shap available")
except Exception as e:
    print(f"[WARN] shap not available: {e}")

# 前端字段名 -> 模型特征名
FIELD_MAPPING = {
    'race': 'Race_Cat',
    'sex': 'Sex',
    'primarySite': 'Location',
    'tStage': 'T_Stage',
    'nStage': 'N_Stage',
    'grade': 'Grade_Cat',
    'age': 'Age_Reclassified',
    'histology': 'Histology',
    # 也支持直接使用模型特征名
    'Race_Cat': 'Race_Cat',
    'Sex': 'Sex',
    'Location': 'Location',
    'T_Stage': 'T_Stage',
    'N_Stage': 'N_Stage',
    'Grade_Cat': 'Grade_Cat',
    'Age_Reclassified': 'Age_Reclassified',
    'Histology': 'Histology',
}

# 前端显示值 -> 模型内部类别值
VALUE_NORMALIZE = {
    'Histology': {
        'Squamous Cell Carcinoma': 'Squamous cell carcinoma',
        'Neuroendocrine Carcinoma': 'Neuroendocrine carcinoma',
        'Other Carcinoma': 'Other',
        'Adenocarcinoma': 'Adenocarcinoma',
    },
}

# 中文标签映射
CHINESE_LABELS = {
    'Race_Cat': '种族',
    'Sex': '性别',
    'Location': '原发部位',
    'T_Stage': 'T分期',
    'N_Stage': 'N分期',
    'Grade_Cat': '分化程度',
    'Age_Reclassified': '年龄分组',
    'Histology': '组织学类型'
}

# 特征选项定义（基于 category_options，去掉 _Cat 后缀）
FEATURE_OPTIONS = {}
for model_key, vals in category_options.items():
    display_key = model_key.replace('_Cat', '')
    if display_key == 'Age_Reclassified':
        display_key = 'Age'
    elif display_key == 'T_Stage':
        display_key = 'T Stage'
    elif display_key == 'N_Stage':
        display_key = 'N Stage'
    FEATURE_OPTIONS[display_key] = vals


# ============ 工具函数 ============

def _build_input_data(data):
    """将前端输入转换为模型可用的 DataFrame"""
    input_dict = {}
    for front_key, model_key in FIELD_MAPPING.items():
        if front_key in data and data[front_key]:
            value = data[front_key]
            # 标准化前端值到模型内部值
            if model_key in VALUE_NORMALIZE and value in VALUE_NORMALIZE[model_key]:
                value = VALUE_NORMALIZE[model_key][value]
            input_dict[model_key] = value

    # 验证是否有所有必需的特征
    required = list(category_options.keys())
    missing_fields = [f for f in required if f not in input_dict]
    if missing_fields:
        raise ValueError(f'缺少必填字段: {", ".join(missing_fields)}')

    # 检查类别有效性
    for col, value in input_dict.items():
        valid = category_options.get(col, [])
        if value not in valid:
            raise ValueError(f'"{col}" 的值 "{value}" 无效，可选: {valid}')

    input_df = pd.DataFrame([input_dict])
    return input_df[required]


def _predict_proba(input_df):
    """返回正类概率（已校准或未经校准）和原始概率"""
    Xt = preprocessor.transform(input_df)
    raw_prob = base_model.predict_proba(Xt)[0, 1]

    if USE_CALIBRATION and calibration_model is not None:
        prob = _apply_calibration(raw_prob)
        # 裁剪到 [0, 1] 防止外推产生越界
        prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    else:
        prob = raw_prob

    return float(prob), float(raw_prob)


def _get_percentile(prob):
    """基于验证集校准概率分布计算百分位"""
    if val_cal_probs is None or len(val_cal_probs) == 0:
        return None
    return float(np.mean(val_cal_probs < prob) * 100)


def _aggregate_shap_values(shap_values_onehot, feature_names_list, cat_cols):
    """将 OneHot 级别的 SHAP 值聚合回原始分类变量"""
    agg = {c: 0.0 for c in cat_cols}
    for fn, sv in zip(feature_names_list, shap_values_onehot):
        # fn like 'cat__T_Stage_T1'
        if fn.startswith('cat__'):
            fn = fn[5:]
        matched = None
        for c in cat_cols:
            if fn.startswith(c + '_'):
                matched = c
                break
        if matched:
            agg[matched] += float(sv)

    # 映射为更友好的显示名
    display_map = {
        'Race_Cat': 'Race', 'Sex': 'Sex', 'Location': 'Location',
        'Histology': 'Histology', 'Grade_Cat': 'Grade',
        'T_Stage': 'T Stage', 'N_Stage': 'N Stage', 'Age_Reclassified': 'Age'
    }
    result = []
    for c, score in agg.items():
        result.append({
            'feature': display_map.get(c, c),
            'contribution': round(score, 6),
            'direction': 'increases' if score > 0 else 'decreases' if score < 0 else 'neutral'
        })
    # 按绝对贡献排序
    result.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return result


def _compute_shap(input_df):
    """计算并返回聚合后的 SHAP 解释"""
    if not HAS_SHAP or shap is None or base_model is None or preprocessor is None:
        return []
    try:
        Xt = preprocessor.transform(input_df)
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(Xt)

        # XGBoost 二分类：shap_values 可能是 [neg, pos] 列表或 (n_samples, n_features, 2) 数组
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[..., 1]
        else:
            sv = shap_values

        sv = np.asarray(sv).reshape(-1)
        cat_cols = list(category_options.keys())
        return _aggregate_shap_values(sv, feature_names, cat_cols)
    except Exception as e:
        print(f"[WARN] SHAP computation failed: {e}")
        return []


# ============ 路由定义 ============

@app.route('/')
def home():
    """渲染主页"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': base_model is not None and preprocessor is not None,
        'calibration_loaded': calibration_model is not None,
        'calibration_enabled': USE_CALIBRATION,
        'shap_available': HAS_SHAP,
        'threshold': float(optimal_threshold),
        'baseline_risk': round(float(BASELINE_RISK) * 100, 2)
    })


@app.route('/api/options', methods=['GET'])
def get_options():
    """返回所有特征的选项"""
    return jsonify({
        'feature_options': FEATURE_OPTIONS,
        'chinese_labels': CHINESE_LABELS,
        'feature_order': [FIELD_MAPPING[k] for k in ['sex', 'race', 'primarySite', 'tStage', 'nStage', 'grade', 'age', 'histology']]
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    return jsonify({
        'features': list(category_options.keys()),
        'display_features': list(FEATURE_OPTIONS.keys()),
        'threshold': float(optimal_threshold),
        'baseline_risk': round(float(BASELINE_RISK) * 100, 2),
        'feature_options': FEATURE_OPTIONS,
        'chinese_labels': CHINESE_LABELS,
        'model_type': 'XGBoost Classifier',
        'calibrated': USE_CALIBRATION and calibration_model is not None,
        'calibration_method': calibration_report.get('calibration_method') if USE_CALIBRATION else None,
        'calibration_comparison': calibration_comparison,
        'shap_available': HAS_SHAP,
        'description': '基于SEER数据库的食管癌肺转移预测模型（当前论文最优模型）',
        'n_samples': calibration_report.get('n_samples'),
        'performance': {
            'val_auc': calibration_report.get('val_auc_calibrated'),
            'test_auc': calibration_report.get('test_auc_calibrated'),
            'brier_test': calibration_report.get('brier_test_calibrated'),
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    if base_model is None or preprocessor is None:
        return jsonify({'error': '模型未加载，请检查服务器配置'}), 500

    try:
        data = request.json
        print(f"收到的原始数据: {data}")

        input_df = _build_input_data(data)
        print(f"转换后的输入数据: {input_df.to_dict()}")

        prob_positive, prob_raw = _predict_proba(input_df)
        prob_negative = 1.0 - prob_positive
        prediction = 1 if prob_positive >= optimal_threshold else 0

        # 风险分层（基于已校准概率分布）
        # 校准后概率：中位数 ~5.9%，P90 ~13.6%，最大值 ~22.4%
        if prob_positive < 0.05:
            risk_level = 'low'
            risk_level_cn = '低风险'
            risk_color = 'green'
        elif prob_positive < 0.15:
            risk_level = 'medium'
            risk_level_cn = '中风险'
            risk_color = 'orange'
        else:
            risk_level = 'high'
            risk_level_cn = '高风险'
            risk_color = 'red'

        # 风险解释
        percentile = _get_percentile(prob_positive)
        relative_risk = prob_positive / BASELINE_RISK if BASELINE_RISK > 0 else None

        # SHAP 解释
        shap_explanation = _compute_shap(input_df)

        result = {
            'success': True,
            'probability': {
                'positive': round(prob_positive * 100, 2),
                'negative': round(prob_negative * 100, 2),
                'raw': round(prob_raw * 100, 2) if USE_CALIBRATION else None,
            },
            'baseline_risk': round(float(BASELINE_RISK) * 100, 2),
            'relative_risk': round(float(relative_risk), 2) if relative_risk is not None else None,
            'percentile': round(float(percentile), 1) if percentile is not None else None,
            'prediction': int(prediction),
            'threshold': round(float(optimal_threshold) * 100, 2),
            'risk_level': str(risk_level),
            'risk_level_cn': str(risk_level_cn),
            'risk_color': str(risk_color),
            'confidence': round(float(max(prob_positive, prob_negative)) * 100, 2),
            'calibrated': USE_CALIBRATION and calibration_model is not None,
            'shap_explanation': shap_explanation,
            'interpretation': f'该患者肺转移概率为 {round(prob_positive * 100, 2)}%，属于{risk_level_cn}。',
            'message': '预测成功'
        }

        result = convert_to_python_type(result)
        print(f"预测结果: {result}")
        return jsonify(result)

    except ValueError as e:
        print(f"输入验证错误: {e}")
        return jsonify({
            'success': False,
            'error': f'输入参数错误: {str(e)}'
        }), 400

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"预测错误: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'预测过程中发生错误: {str(e)}'
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    if base_model is None or preprocessor is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        data = request.json
        patients = data.get('patients', [])

        if not patients:
            return jsonify({'error': '没有提供患者数据'}), 400

        results = []
        for i, patient in enumerate(patients):
            try:
                input_df = _build_input_data(patient)
                prob_positive, prob_raw = _predict_proba(input_df)
                percentile = _get_percentile(prob_positive)
                relative_risk = prob_positive / BASELINE_RISK if BASELINE_RISK > 0 else None

                results.append({
                    'index': int(i),
                    'probability': round(prob_positive * 100, 2),
                    'probability_raw': round(prob_raw * 100, 2) if USE_CALIBRATION else None,
                    'baseline_risk': round(float(BASELINE_RISK) * 100, 2),
                    'relative_risk': round(float(relative_risk), 2) if relative_risk is not None else None,
                    'percentile': round(float(percentile), 1) if percentile is not None else None,
                    'prediction': int(1 if prob_positive >= optimal_threshold else 0),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'index': int(i),
                    'error': str(e),
                    'success': False
                })

        results = convert_to_python_type(results)
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    print(f"\n{'='*50}")
    print(f"启动食管癌肺转移预测服务器")
    print(f"模型: XGBoost {'(已校准)' if USE_CALIBRATION else '(未校准)'}")
    print(f"校准方法: {calibration_report.get('calibration_method', 'N/A') if USE_CALIBRATION else 'None'}")
    print(f"阈值: {optimal_threshold}")
    print(f"SHAP: {'可用' if HAS_SHAP else '不可用'}")
    print(f"端口: {port}")
    print(f"调试模式: {debug}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
