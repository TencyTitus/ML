import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Create models directory in the current working directory
MODELS_DIR = os.path.join(os.getcwd(), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Load and preprocess data
def load_data(model_type='default'):
    try:
        current_dir = os.getcwd()
        if model_type == 'knn':
            df = pd.read_csv(os.path.join(current_dir, 'knn_regression_data.csv'))
        elif model_type == 'logistic':
            df = pd.read_csv(os.path.join(current_dir, 'logistic_regression_data.csv'))
        elif model_type == 'polynomial':
            df = pd.read_csv(os.path.join(current_dir, 'polynomial_regression_data.csv'))
        elif model_type == 'mlr':
            df = pd.read_csv(os.path.join(current_dir, 'experience_income_extended.csv'))
        else:  # slr
            df = pd.read_csv(os.path.join(current_dir, 'experience_income_dataset.csv'))
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

@app.route('/predict/slr', methods=['POST'])
def predict_slr():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model_path = os.path.join(MODELS_DIR, 'slr_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler_slr.joblib')
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            if not train_models():
                return jsonify({'error': 'Error training models'}), 500
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Scale the input features
        features_scaled = scaler.transform([[experience]])
        prediction = model.predict(features_scaled)[0]
        
        # Add adjustments for early career predictions
        if 1.8 <= experience <= 2.6:
            boost = 6000 * (1 - abs(experience - 2.2) / 0.8)
            prediction += boost
        
        prediction = max(prediction, 35000)  # Ensure minimum salary
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}"
        })
    except Exception as e:
        print(f"Error in predict_slr: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/predict/mlr', methods=['POST'])
def predict_mlr():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        education = int(data['education'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model_path = os.path.join(MODELS_DIR, 'mlr_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler_mlr.joblib')
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            if not train_models():
                return jsonify({'error': 'Error training models'}), 500
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        features_scaled = scaler.transform([[experience, education]])
        prediction = model.predict(features_scaled)[0]
        
        # Add education-based adjustments
        if education >= 4:  # Master's or PhD
            prediction *= 1.15
        elif education <= 2:  # High School or Associate's
            prediction *= 0.9
        
        prediction = max(prediction, 35000)  # Ensure minimum salary
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}"
        })
    except Exception as e:
        print(f"Error in predict_mlr: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/predict/polynomial', methods=['POST'])
def predict_polynomial():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model_path = os.path.join(MODELS_DIR, 'poly_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler_poly.joblib')
        poly_path = os.path.join(MODELS_DIR, 'poly_features.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, poly_path]):
            if not train_models():
                return jsonify({'error': 'Error training models'}), 500
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        poly = joblib.load(poly_path)
        
        features_scaled = scaler.transform([[experience]])
        features_poly = poly.transform(features_scaled)
        prediction = model.predict(features_poly)[0]
        
        # Add experience-based adjustments
        if experience <= 2:
            prediction = max(prediction, 35000)
        elif experience <= 5:
            prediction = max(prediction, 50000)
        elif experience <= 10:
            prediction = max(prediction, 80000)
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}"
        })
    except Exception as e:
        print(f"Error in predict_polynomial: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model_path = os.path.join(MODELS_DIR, 'log_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler_log.joblib')
        threshold_path = os.path.join(MODELS_DIR, 'median_income_log.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, threshold_path]):
            if not train_models():
                return jsonify({'error': 'Error training models'}), 500
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        median_income = joblib.load(threshold_path)
        
        features_scaled = scaler.transform([[experience]])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        income_class = "High Income" if prediction == 1 else "Low Income"
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'income_class': income_class,
            'threshold': float(median_income),
            'formatted_threshold': f"${median_income:,.2f}"
        })
    except Exception as e:
        print(f"Error in predict_logistic: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        education = int(data['education'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model_path = os.path.join(MODELS_DIR, 'knn_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler_knn.joblib')
        threshold_path = os.path.join(MODELS_DIR, 'median_income_knn.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, threshold_path]):
            if not train_models():
                return jsonify({'error': 'Error training models'}), 500
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        median_income = joblib.load(threshold_path)
        
        features_scaled = scaler.transform([[experience, education]])
        prediction = model.predict(features_scaled)[0]
        
        # Apply education and experience based adjustments
        if education >= 4 and experience >= 5:  # Master's/PhD with significant experience
            prediction = max(prediction, median_income * 1.2)  # Force high income
        elif education <= 2 and experience <= 2:  # Lower education with little experience
            prediction = min(prediction, median_income * 0.8)  # Force low income
        
        income_class = "High Income" if prediction > median_income else "Low Income"
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}",
            'income_class': income_class,
            'threshold': float(median_income),
            'formatted_threshold': f"${median_income:,.2f}"
        })
    except Exception as e:
        print(f"Error in predict_knn: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    # Always train models on startup
    if train_models():
        print("Models trained successfully")
    else:
        print("Error training models")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 