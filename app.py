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


# ============ 值映射：前端显示值 -> 编码器数字字符串 ============
VALUE_MAPPING = {
    'Race': {
        'White': '0',
        'Black': '1',
        'Other': '2'
        # 注意：编码器有 0,1,2,3 四个值，如果有第四个种族类别需要添加
    },
    'Sex': {
        'Male': '0',
        'Female': '1'
    },
    'Primary site': {
        'Upper third': '0',
        'Middle third': '1',
        'Lower third': '2',
        'Overlapping': '3'
        # 注意：编码器有 0,1,2,3,4 五个值，如果有第五个部位需要添加
    },
    'T stage': {
        'T1': '0',
        'T2': '1',
        'T3': '2',
        'T4': '3'
    },
    'N stage': {
        'N0': '0',
        'N1': '1',
        'N2': '2',
        'N3': '3'
    },
    'Grade': {
        'Grade I': '0',
        'Grade II': '1',
        'Grade III': '2',
        'Grade IV': '3'
    },
    'Age_Reclassified': {
        '<50': '0',
        '50-59': '1',
        '60-69': '2',
        '70-79': '3'
        # 注意：你的 FEATURE_OPTIONS 有 '>=80'，但编码器只有 0,1,2,3
        # 如果需要 >=80，可能要检查原始数据
    },
    'Histology': {
        'Adenocarcinoma': '0',
        'Squamous cell carcinoma': '1',
        'Other': '2'
        # 注意：编码器有 0,1,2,3 四个值，如果有第四个组织学类型需要添加
    }
}


def map_input_value(col, value):
    """将前端值映射到编码器期望的值"""
    if col in VALUE_MAPPING and value in VALUE_MAPPING[col]:
        return VALUE_MAPPING[col][value]
    return value


# ============ 加载模型和配置文件 ============
print("正在加载模型文件...")

try:
    model = joblib.load('esophageal_cancer_rf_model.pkl')
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    model = None

try:
    encoders = joblib.load('label_encoders.pkl')
    print("✓ 编码器加载成功")
except Exception as e:
    print(f"✗ 编码器加载失败: {e}")
    encoders = {}

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

# ============ 特征选项定义（前端显示用）============
FEATURE_OPTIONS = {
    'Race': ['White', 'Black', 'Other'],
    'Sex': ['Male', 'Female'],
    'Primary site': ['Upper third', 'Middle third', 'Lower third', 'Overlapping'],
    'T stage': ['T1', 'T2', 'T3', 'T4'],
    'N stage': ['N0', 'N1', 'N2', 'N3'],
    'Grade': ['Grade I', 'Grade II', 'Grade III', 'Grade IV'],
    'Age_Reclassified': ['<50', '50-59', '60-69', '70-79'],
    'Histology': ['Adenocarcinoma', 'Squamous cell carcinoma', 'Other']
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
        'encoders_loaded': len(encoders) > 0,
        'threshold': float(optimal_threshold)
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    return jsonify({
        'features': [str(f) for f in feature_names],
        'threshold': float(optimal_threshold),
        'feature_options': FEATURE_OPTIONS,
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

        # 验证输入数据
        required_fields = ['race', 'sex', 'primarySite', 'tStage', 'nStage', 'grade', 'age', 'histology']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'缺少必填字段: {field}'}), 400

        # 创建输入数据框，进行值映射并直接转为整数
        input_data = pd.DataFrame([{
            'Race': int(map_input_value('Race', data['race'])),
            'Sex': int(map_input_value('Sex', data['sex'])),
            'Primary site': int(map_input_value('Primary site', data['primarySite'])),
            'T stage': int(map_input_value('T stage', data['tStage'])),
            'N stage': int(map_input_value('N stage', data['nStage'])),
            'Grade': int(map_input_value('Grade', data['grade'])),
            'Age_Reclassified': int(map_input_value('Age_Reclassified', data['age'])),
            'Histology': int(map_input_value('Histology', data['histology']))
        }])

        print(f"输入数据: {input_data.to_dict()}")

        # 确保特征顺序与训练时一致
        try:
            input_data = input_data[feature_names]
        except KeyError as e:
            return jsonify({'error': f'特征名称不匹配: {e}'}), 400

        # 进行预测（不再需要编码器转换）
        prob_array = model.predict_proba(input_data)[0]
        probability = float(prob_array[1])
        prediction = 1 if probability >= optimal_threshold else 0

        # 确定风险等级
        if probability < 0.1:
            risk_level = 'low'
        elif probability < 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        confidence = float(max(probability, 1.0 - probability))

        result = {
            'probability': float(probability),
            'prediction': int(prediction),
            'threshold': float(optimal_threshold),
            'risk_level': str(risk_level),
            'confidence': float(confidence),
            'message': '预测成功'
        }

        result = convert_to_python_type(result)
        print(f"预测结果: {result}")
        return jsonify(result)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"预测错误: {error_trace}")
        return jsonify({'error': f'预测过程中发生错误: {str(e)}'}), 500


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

        results = []
        for i, patient in enumerate(patients):
            try:
                input_data = pd.DataFrame([{
                    'Race': map_input_value('Race', patient['race']),
                    'Sex': map_input_value('Sex', patient['sex']),
                    'Primary site': map_input_value('Primary site', patient['primarySite']),
                    'T stage': map_input_value('T stage', patient['tStage']),
                    'N stage': map_input_value('N stage', patient['nStage']),
                    'Grade': map_input_value('Grade', patient['grade']),
                    'Age_Reclassified': map_input_value('Age_Reclassified', patient['age']),
                    'Histology': map_input_value('Histology', patient['histology'])
                }])

                for col in input_data.columns:
                    if col in encoders:
                        le = encoders[col]
                        if input_data[col].values[0] in le.classes_:
                            input_data[col] = le.transform(input_data[col])

                input_data = input_data[feature_names]
                prob_array = model.predict_proba(input_data)[0]
                probability = float(prob_array[1])

                results.append({
                    'index': int(i),
                    'probability': float(probability),
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
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"启动服务器，端口: {port}, 调试模式: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
