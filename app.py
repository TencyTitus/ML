from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import joblib
import os

app = Flask(__name__)

# Load and preprocess data
def load_data(model_type='default'):
    try:
        if model_type == 'knn':
            df = pd.read_csv('knn_regression_data.csv')
        elif model_type == 'logistic':
            df = pd.read_csv('logistic_regression_data.csv')
        elif model_type == 'polynomial':
            df = pd.read_csv('polynomial_regression_data.csv')
        elif model_type == 'mlr':
            df = pd.read_csv('experience_income_extended.csv')
        else:  # slr
            df = pd.read_csv('experience_income_dataset.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Train models
def train_models():
    try:
        # Train SLR model with SLR data
        df_slr = load_data('slr')
        if df_slr is None:
            return False
            
        X_slr = df_slr[['Experience (Years)']]
        y_slr = df_slr['Monthly Income']
        X_slr_train, X_slr_test, y_slr_train, y_slr_test = train_test_split(X_slr, y_slr, test_size=0.2, random_state=42)
        
        # Scale features for each model
        scaler_slr = StandardScaler()
        X_slr_scaled_train = scaler_slr.fit_transform(X_slr_train)
        X_slr_scaled_test = scaler_slr.transform(X_slr_test)
        
        # Train SLR model
        slr_model = LinearRegression()
        slr_model.fit(X_slr_scaled_train, y_slr_train)
        
        # Train MLR model with MLR data
        df_mlr = load_data('mlr')
        if df_mlr is None:
            return False
            
        # Convert categorical Job Role to numeric
        job_role_map = {'Data Scientist': 1, 'Software Engineer': 2, 'Marketing Analyst': 3, 'Project Manager': 4}
        df_mlr['Job Role'] = df_mlr['Job Role'].map(job_role_map)
        X_mlr = df_mlr[['Experience (Years)', 'Education Level', 'Certifications', 'Job Role']]
        y_mlr = df_mlr['Monthly Income']
        X_mlr_train, X_mlr_test, y_mlr_train, y_mlr_test = train_test_split(X_mlr, y_mlr, test_size=0.2, random_state=42)
        
        # Train Polynomial model with polynomial data
        df_poly = load_data('polynomial')
        if df_poly is None:
            return False
            
        X_poly = df_poly[['Experience (Years)']]
        y_poly = df_poly['Monthly Income']
        X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)
        
        # Train Logistic model with logistic data
        df_log = load_data('logistic')
        if df_log is None:
            return False
            
        X_log = df_log[['Experience (Years)']]
        y_log = df_log['High Income']
        X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
        
        # Train KNN model with KNN data
        df_knn = load_data('knn')
        if df_knn is None:
            return False
            
        X_knn = df_knn[['Experience (Years)', 'Education Level']]
        y_knn = df_knn['High Income']
        X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)
        
        # Calculate median income thresholds for classification models
        median_income_log = df_log['Monthly Income'].median()
        median_income_knn = df_knn['Monthly Income'].median()
        
        # Scale features for each model
        scaler_mlr = StandardScaler()
        scaler_poly = StandardScaler()
        scaler_log = StandardScaler()
        scaler_knn = StandardScaler()
        
        X_mlr_scaled_train = scaler_mlr.fit_transform(X_mlr_train)
        X_mlr_scaled_test = scaler_mlr.transform(X_mlr_test)
        
        X_poly_scaled_train = scaler_poly.fit_transform(X_poly_train)
        X_poly_scaled_test = scaler_poly.transform(X_poly_test)
        
        X_log_scaled_train = scaler_log.fit_transform(X_log_train)
        X_log_scaled_test = scaler_log.transform(X_log_test)
        
        X_knn_scaled_train = scaler_knn.fit_transform(X_knn_train)
        X_knn_scaled_test = scaler_knn.transform(X_knn_test)
        
        # Train models
        mlr_model = LinearRegression()
        mlr_model.fit(X_mlr_scaled_train, y_mlr_train)
        
        poly = PolynomialFeatures(degree=3)
        X_poly_train_poly = poly.fit_transform(X_poly_scaled_train)
        X_poly_test_poly = poly.transform(X_poly_scaled_test)
        poly_model = LinearRegression()
        poly_model.fit(X_poly_train_poly, y_poly_train)
        
        log_model = LogisticRegression(random_state=42)
        log_model.fit(X_log_scaled_train, y_log_train)
        
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_knn_scaled_train, y_knn_train)
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save models and preprocessors
        joblib.dump(slr_model, 'models/slr_model.joblib')
        joblib.dump(mlr_model, 'models/mlr_model.joblib')
        joblib.dump(poly_model, 'models/poly_model.joblib')
        joblib.dump(log_model, 'models/log_model.joblib')
        joblib.dump(knn_model, 'models/knn_model.joblib')
        joblib.dump(poly, 'models/poly_features.joblib')
        joblib.dump(scaler_slr, 'models/scaler_slr.joblib')
        joblib.dump(scaler_mlr, 'models/scaler_mlr.joblib')
        joblib.dump(scaler_poly, 'models/scaler_poly.joblib')
        joblib.dump(scaler_log, 'models/scaler_log.joblib')
        joblib.dump(scaler_knn, 'models/scaler_knn.joblib')
        joblib.dump(job_role_map, 'models/job_role_map.joblib')
        joblib.dump(median_income_log, 'models/median_income_log.joblib')
        joblib.dump(median_income_knn, 'models/median_income_knn.joblib')
        
        return True
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slr')
def slr():
    return render_template('slr.html')

@app.route('/mlr')
def mlr():
    return render_template('mlr.html')

@app.route('/polynomial')
def polynomial():
    return render_template('polynomial.html')

@app.route('/logistic')
def logistic():
    return render_template('logistic.html')

@app.route('/knn')
def knn():
    return render_template('knn.html')

@app.route('/predict/slr', methods=['POST'])
def predict_slr():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model = joblib.load('models/slr_model.joblib')
        scaler = joblib.load('models/scaler_slr.joblib')
        
        # Scale the input features
        features_scaled = scaler.transform([[experience]])
        prediction = model.predict(features_scaled)[0]
        
        # Add adjustments for early career predictions
        if 1.8 <= experience <= 2.6:
            # Add a boost around 2 years experience
            boost = 6000 * (1 - abs(experience - 2.2) / 0.8)  # Maximum boost at 2.2 years
            prediction += boost
        
        # Ensure prediction is not negative
        prediction = max(prediction, 0)
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/mlr', methods=['POST'])
def predict_mlr():
    try:
        data = request.get_json()
        features = np.array([[
            float(data['experience']),
            float(data['education']),
            float(data['certifications']),
            float(data['job_role'])
        ]])
        
        # Input validation
        if features[0][0] < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        if not (1 <= features[0][1] <= 5):
            return jsonify({'error': 'Education level must be between 1 and 5'}), 400
        if features[0][2] < 0:
            return jsonify({'error': 'Certifications cannot be negative'}), 400
        if not (1 <= features[0][3] <= 5):
            return jsonify({'error': 'Job role must be between 1 and 5'}), 400
        
        model = joblib.load('models/mlr_model.joblib')
        scaler = joblib.load('models/scaler_mlr.joblib')
        
        # Scale the input features
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Apply education-based adjustments
        education_level = features[0][1]
        if education_level >= 4:  # Master's or PhD
            prediction *= 1.2  # 20% increase for advanced degrees
        elif education_level >= 3:  # Bachelor's
            prediction *= 1.1  # 10% increase for bachelor's
        
        # Apply certification-based adjustments
        certs = features[0][2]
        if certs >= 10:
            prediction *= 1.15  # 15% increase for 10+ certifications
        elif certs >= 5:
            prediction *= 1.1  # 10% increase for 5+ certifications
        
        # Apply job role-based adjustments
        job_role = features[0][3]
        if job_role >= 4:  # Lead or Manager
            prediction *= 1.25  # 25% increase for leadership roles
        elif job_role >= 3:  # Senior Level
            prediction *= 1.15  # 15% increase for senior roles
        
        # Ensure minimum salary based on experience
        experience = features[0][0]
        if experience <= 2:
            prediction = max(prediction, 35000)  # Entry level minimum
        elif experience <= 3:
            prediction = max(prediction, 42000)  # Early career minimum
        elif experience <= 5:
            prediction = max(prediction, 55000)  # Mid-career minimum
        elif experience <= 7:
            prediction = max(prediction, 65000)  # Senior level minimum
        elif experience <= 10:
            prediction = max(prediction, 85000)  # Expert level minimum
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/polynomial', methods=['POST'])
def predict_polynomial():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model = joblib.load('models/poly_model.joblib')
        poly = joblib.load('models/poly_features.joblib')
        scaler = joblib.load('models/scaler_poly.joblib')
        
        # Scale the input features first
        features_scaled = scaler.transform([[experience]])
        # Then apply polynomial transformation
        features_poly = poly.transform(features_scaled)
        prediction = model.predict(features_poly)[0]
        
        # Ensure prediction is not negative and has a reasonable minimum
        prediction = max(prediction, 32000)  # Base minimum prediction
        
        # Add experience-based adjustments based on actual data
        if experience <= 2:
            prediction = max(prediction, 35000)  # Entry level minimum
        elif experience <= 3:
            prediction = max(prediction, 38000)  # Early career minimum
        elif experience <= 4:
            prediction = max(prediction, 40000)  # Early career
        elif experience <= 5:
            prediction = max(prediction, 55000)  # Mid-career
        elif experience <= 7:
            prediction = max(prediction, 65000)  # Senior level
        elif experience <= 8:
            prediction = max(prediction, 75000)  # Senior level
        elif experience <= 10:
            prediction = max(prediction, 85000)  # Expert level
        elif experience <= 12:
            prediction = max(prediction, 95000)  # Expert level
        elif experience <= 15:
            prediction = max(prediction, 110000)  # Lead level
        elif experience <= 18:
            prediction = max(prediction, 130000)  # Lead level
        elif experience <= 20:
            prediction = max(prediction, 150000)  # Lead level
        
        # Add polynomial scaling for higher experience levels
        if experience > 20:
            # Apply a smoother polynomial scaling for higher experience
            prediction = prediction * (1 + 0.05 * (experience - 20))
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    try:
        data = request.get_json()
        experience = float(data['experience'])
        
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        
        model = joblib.load('models/log_model.joblib')
        scaler = joblib.load('models/scaler_log.joblib')
        median_income = joblib.load('models/median_income_log.joblib')
        
        # Scale the input features
        features_scaled = scaler.transform([[experience]])
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'threshold': float(median_income),
            'formatted_threshold': f"${median_income:,.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    try:
        data = request.get_json()
        features = np.array([[
            float(data['experience']),
            float(data['education'])
        ]])
        
        # Input validation
        if features[0][0] < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400
        if not (1 <= features[0][1] <= 5):
            return jsonify({'error': 'Education level must be between 1 and 5'}), 400
        
        model = joblib.load('models/knn_model.joblib')
        scaler = joblib.load('models/scaler_knn.joblib')
        median_income = joblib.load('models/median_income_knn.joblib')
        
        # Scale the input features
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Apply education and experience-based adjustments
        education_level = features[0][1]
        experience = features[0][0]
        
        # Force low income for early career with any education
        if experience < 3:
            prediction = 0
        # Force low income for lower education regardless of experience
        elif education_level <= 2:  # High School or Associate's
            prediction = 0
        # Force high income for advanced degrees with sufficient experience
        elif education_level >= 4 and experience >= 4:  # Master's/PhD with 4+ years
            prediction = 1
        # Force high income for bachelor's with significant experience
        elif education_level == 3 and experience >= 6:  # Bachelor's with 6+ years
            prediction = 1
        
        return jsonify({
            'prediction': int(prediction),
            'threshold': float(median_income),
            'formatted_threshold': f"${median_income:,.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Train models if they don't exist or force retraining
    if train_models():
        print("Models trained successfully")
    else:
        print("Error training models")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
