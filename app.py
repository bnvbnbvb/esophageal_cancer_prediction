from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)


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


# ============ 手动编码映射（替代 label_encoders.pkl）============
MANUAL_ENCODERS = {
    'Race': {
        'American Indian/Alaska Native': 0,
        'Asian or Pacific Islander': 1,
        'Black': 2,
        'White': 3,
    },
    'Sex': {
        'Female': 0,
        'Male': 1,
    },
    'Primary site': {
        'Abdominal/overlapping esophagus': 0,
        'Cervical esophagus': 1,
        'Lower third of esophagus': 2,
        'Middle third of esophagus': 3,
        'Upper third of esophagus': 4,
    },
    'T stage': {
        'T1': 0,
        'T2': 1,
        'T3': 2,
        'T4': 3,
    },
    'N stage': {
        'N0': 0,
        'N1': 1,
        'N2': 2,
        'N3': 3,
    },
    'Grade': {
        'Moderately differentiated; Grade II': 0,
        'Poorly differentiated; Grade III': 1,
        'Undifferentiated; anaplastic; Grade IV': 2,
        'Well differentiated; Grade I': 3,
    },
    'Age_Reclassified': {
        '<50': 0,
        '50-60': 1,
        '60-70': 2,
        '70-80': 3,
        '≥80': 4,
    },
    'Histology': {
        'Adenocarcinoma': 0,
        'Neuroendocrine Carcinoma': 1,
        'Other Carcinoma': 2,
        'Squamous Cell Carcinoma': 3,
    },
}


# ============ 加载模型和配置文件 ============
print("正在加载模型文件...")

try:
    model = joblib.load('esophageal_cancer_rf_model.joblib')
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    model = None

try:
    feature_names = joblib.load('feature_names.pkl')
    feature_names = [str(f) for f in feature_names]
    print(f"✓ 特征名称加载成功: {feature_names}")
except Exception as e:
    print(f"✗ 特征名称加载失败: {e}")
    feature_names = ['Race', 'Sex', 'Primary site', 'T stage', 'N stage', 'Grade', 'Age_Reclassified', 'Histology']

try:
    optimal_threshold = joblib.load('optimal_threshold.pkl')
    optimal_threshold = float(optimal_threshold)
    print(f"✓ 最佳阈值加载成功: {optimal_threshold}")
except Exception as e:
    print(f"✗ 阈值加载失败，使用默认值0.5: {e}")
    optimal_threshold = 0.5

print("✓ 使用手动编码映射")


# ============ 特征选项定义（基于 MANUAL_ENCODERS 的键）============
FEATURE_OPTIONS = {
    'Race': ['White', 'Black', 'Asian or Pacific Islander', 'American Indian/Alaska Native'],
    'Sex': ['Male', 'Female'],
    'Primary site': [
        'Cervical esophagus',
        'Upper third of esophagus',
        'Middle third of esophagus',
        'Lower third of esophagus',
        'Abdominal/overlapping esophagus'
    ],
    'T stage': ['T1', 'T2', 'T3', 'T4'],
    'N stage': ['N0', 'N1', 'N2', 'N3'],
    'Grade': [
        'Well differentiated; Grade I',
        'Moderately differentiated; Grade II',
        'Poorly differentiated; Grade III',
        'Undifferentiated; anaplastic; Grade IV'
    ],
    'Age_Reclassified': ['<50', '50-60', '60-70', '70-80', '≥80'],
    'Histology': ['Squamous Cell Carcinoma', 'Adenocarcinoma', 'Neuroendocrine Carcinoma', 'Other Carcinoma']
}

# 中文标签映射
CHINESE_LABELS = {
    'Race': '种族',
    'Sex': '性别',
    'Primary site': '原发部位',
    'T stage': 'T分期',
    'N stage': 'N分期',
    'Grade': '分化程度',
    'Age_Reclassified': '年龄分组',
    'Histology': '组织学类型'
}


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
        'model_loaded': model is not None,
        'encoders_loaded': True,
        'threshold': float(optimal_threshold)
    })


@app.route('/api/options', methods=['GET'])
def get_options():
    """返回所有特征的选项"""
    return jsonify({
        'feature_options': FEATURE_OPTIONS,
        'chinese_labels': CHINESE_LABELS,
        'feature_order': feature_names
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    return jsonify({
        'features': [str(f) for f in feature_names],
        'threshold': float(optimal_threshold),
        'feature_options': FEATURE_OPTIONS,
        'chinese_labels': CHINESE_LABELS,
        'model_type': 'Random Forest Classifier',
        'description': '基于SEER数据库的食管癌肺转移预测模型'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    if model is None:
        return jsonify({'error': '模型未加载，请检查服务器配置'}), 500

    try:
        data = request.json
        print(f"收到的原始数据: {data}")

        # 字段映射：前端字段名 -> 模型特征名
        field_mapping = {
            'race': 'Race',
            'sex': 'Sex',
            'primarySite': 'Primary site',
            'tStage': 'T stage',
            'nStage': 'N stage',
            'grade': 'Grade',
            'age': 'Age_Reclassified',
            'histology': 'Histology',
            # 也支持直接使用模型特征名
            'Race': 'Race',
            'Sex': 'Sex',
            'Primary site': 'Primary site',
            'T stage': 'T stage',
            'N stage': 'N stage',
            'Grade': 'Grade',
            'Age_Reclassified': 'Age_Reclassified',
            'Histology': 'Histology'
        }

        # 构建输入数据字典
        input_dict = {}
        for front_key, model_key in field_mapping.items():
            if front_key in data and data[front_key]:
                input_dict[model_key] = data[front_key]

        # 验证是否有所有必需的特征
        missing_fields = [f for f in feature_names if f not in input_dict]
        if missing_fields:
            return jsonify({
                'error': f'缺少必填字段: {", ".join(missing_fields)}',
                'received_fields': list(input_dict.keys()),
                'required_fields': feature_names
            }), 400

        # 创建 DataFrame
        input_data = pd.DataFrame([input_dict])
        print(f"转换后的输入数据: {input_data.to_dict()}")

        # 使用手动编码映射进行标签编码
        for col in feature_names:
            if col in MANUAL_ENCODERS:
                value = input_data[col].values[0]
                
                # 检查值是否在编码器的已知类别中
                if value not in MANUAL_ENCODERS[col]:
                    return jsonify({
                        'error': f'"{col}" 的值 "{value}" 无效',
                        'valid_options': list(MANUAL_ENCODERS[col].keys())
                    }), 400
                
                encoded_value = MANUAL_ENCODERS[col][value]
                input_data[col] = encoded_value
                print(f"  {col}: '{value}' -> {encoded_value}")

        # 确保特征顺序与训练时一致
        input_data = input_data[feature_names]
        print(f"最终输入特征: {input_data.values}")

        # 进行预测
        prob_array = model.predict_proba(input_data)[0]
        prob_negative = float(prob_array[0])
        prob_positive = float(prob_array[1])
        prediction = 1 if prob_positive >= optimal_threshold else 0

        # 确定风险等级
        if prob_positive < 0.2:
            risk_level = 'low'
            risk_level_cn = '低风险'
            risk_color = 'green'
        elif prob_positive < 0.5:
            risk_level = 'medium'
            risk_level_cn = '中风险'
            risk_color = 'orange'
        else:
            risk_level = 'high'
            risk_level_cn = '高风险'
            risk_color = 'red'

        result = {
            'success': True,
            'probability': {
                'positive': round(prob_positive * 100, 2),
                'negative': round(prob_negative * 100, 2)
            },
            'prediction': int(prediction),
            'threshold': round(float(optimal_threshold) * 100, 2),
            'risk_level': str(risk_level),
            'risk_level_cn': str(risk_level_cn),
            'risk_color': str(risk_color),
            'confidence': round(float(max(prob_positive, prob_negative)) * 100, 2),
            'interpretation': f'该患者肺转移概率为 {round(prob_positive * 100, 2)}%，属于{risk_level_cn}。',
            'message': '预测成功'
        }

        result = convert_to_python_type(result)
        print(f"预测结果: {result}")
        return jsonify(result)

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
    if model is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        data = request.json
        patients = data.get('patients', [])

        if not patients:
            return jsonify({'error': '没有提供患者数据'}), 400

        field_mapping = {
            'race': 'Race',
            'sex': 'Sex',
            'primarySite': 'Primary site',
            'tStage': 'T stage',
            'nStage': 'N stage',
            'grade': 'Grade',
            'age': 'Age_Reclassified',
            'histology': 'Histology'
        }

        results = []
        for i, patient in enumerate(patients):
            try:
                # 构建输入数据
                input_dict = {}
                for front_key, model_key in field_mapping.items():
                    if front_key in patient:
                        input_dict[model_key] = patient[front_key]

                input_data = pd.DataFrame([input_dict])

                # 使用手动编码映射
                for col in feature_names:
                    if col in MANUAL_ENCODERS:
                        value = input_data[col].values[0]
                        if value in MANUAL_ENCODERS[col]:
                            input_data[col] = MANUAL_ENCODERS[col][value]
                        else:
                            raise ValueError(f'"{col}" 的值 "{value}" 无效')

                input_data = input_data[feature_names]
                prob_array = model.predict_proba(input_data)[0]
                probability = float(prob_array[1])

                results.append({
                    'index': int(i),
                    'probability': round(probability * 100, 2),
                    'prediction': int(1 if probability >= optimal_threshold else 0),
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
    print(f"端口: {port}")
    print(f"调试模式: {debug}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
