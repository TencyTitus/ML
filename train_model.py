import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Create models directory
MODELS_DIR = os.path.join(os.getcwd(), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def train_models():
    try:
        # Train Simple Linear Regression
        df_slr = pd.read_csv('experience_income_dataset.csv')
        X_slr = df_slr[['Years of Experience']].values
        y_slr = df_slr['Monthly Income'].values
        
        scaler_slr = StandardScaler()
        X_slr_scaled = scaler_slr.fit_transform(X_slr)
        
        model_slr = LinearRegression()
        model_slr.fit(X_slr_scaled, y_slr)
        
        joblib.dump(model_slr, os.path.join(MODELS_DIR, 'slr_model.joblib'))
        joblib.dump(scaler_slr, os.path.join(MODELS_DIR, 'scaler_slr.joblib'))
        
        # Train Multiple Linear Regression
        df_mlr = pd.read_csv('experience_income_extended.csv')
        X_mlr = df_mlr[['Years of Experience', 'Education Level']].values
        y_mlr = df_mlr['Monthly Income'].values
        
        scaler_mlr = StandardScaler()
        X_mlr_scaled = scaler_mlr.fit_transform(X_mlr)
        
        model_mlr = LinearRegression()
        model_mlr.fit(X_mlr_scaled, y_mlr)
        
        joblib.dump(model_mlr, os.path.join(MODELS_DIR, 'mlr_model.joblib'))
        joblib.dump(scaler_mlr, os.path.join(MODELS_DIR, 'scaler_mlr.joblib'))
        
        # Train Polynomial Regression
        df_poly = pd.read_csv('polynomial_regression_data.csv')
        X_poly = df_poly[['Years of Experience']].values
        y_poly = df_poly['Monthly Income'].values
        
        scaler_poly = StandardScaler()
        X_poly_scaled = scaler_poly.fit_transform(X_poly)
        
        poly = PolynomialFeatures(degree=3)
        X_poly_features = poly.fit_transform(X_poly_scaled)
        
        model_poly = LinearRegression()
        model_poly.fit(X_poly_features, y_poly)
        
        joblib.dump(model_poly, os.path.join(MODELS_DIR, 'poly_model.joblib'))
        joblib.dump(scaler_poly, os.path.join(MODELS_DIR, 'scaler_poly.joblib'))
        joblib.dump(poly, os.path.join(MODELS_DIR, 'poly_features.joblib'))
        
        # Train Logistic Regression
        df_log = pd.read_csv('logistic_regression_data.csv')
        X_log = df_log[['Years of Experience']].values
        median_income_log = df_log['Monthly Income'].median()
        y_log = (df_log['Monthly Income'] > median_income_log).astype(int)
        
        scaler_log = StandardScaler()
        X_log_scaled = scaler_log.fit_transform(X_log)
        
        model_log = LogisticRegression()
        model_log.fit(X_log_scaled, y_log)
        
        joblib.dump(model_log, os.path.join(MODELS_DIR, 'log_model.joblib'))
        joblib.dump(scaler_log, os.path.join(MODELS_DIR, 'scaler_log.joblib'))
        joblib.dump(median_income_log, os.path.join(MODELS_DIR, 'median_income_log.joblib'))
        
        # Train KNN
        df_knn = pd.read_csv('knn_regression_data.csv')
        X_knn = df_knn[['Years of Experience', 'Education Level']].values
        y_knn = df_knn['Monthly Income'].values
        median_income_knn = df_knn['Monthly Income'].median()
        
        scaler_knn = StandardScaler()
        X_knn_scaled = scaler_knn.fit_transform(X_knn)
        
        model_knn = KNeighborsRegressor(n_neighbors=5)
        model_knn.fit(X_knn_scaled, y_knn)
        
        joblib.dump(model_knn, os.path.join(MODELS_DIR, 'knn_model.joblib'))
        joblib.dump(scaler_knn, os.path.join(MODELS_DIR, 'scaler_knn.joblib'))
        joblib.dump(median_income_knn, os.path.join(MODELS_DIR, 'median_income_knn.joblib'))
        
        return True
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return False

if __name__ == '__main__':
    if train_models():
        print("Models trained successfully")
    else:
        print("Error training models") 