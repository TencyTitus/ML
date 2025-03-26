# Income Prediction Models

This project implements various regression models to predict monthly income based on different features:

1. Simple Linear Regression (SLR)
   - Predicts income based on years of experience

2. Multiple Linear Regression (MLR)
   - Features: Experience, Education Level, Certifications, Job Role
   - Includes adjustments for education and certification levels

3. Polynomial Regression
   - Non-linear relationship between experience and income
   - Includes experience-based minimum thresholds

4. Logistic Regression
   - Classifies income as high/low based on experience
   - Uses median income as threshold

5. K-Nearest Neighbors (KNN)
   - Classifies income based on experience and education
   - Includes education-based adjustments

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python app.py`

## API Endpoints
- `/predict/slr` - Simple Linear Regression predictions
- `/predict/mlr` - Multiple Linear Regression predictions
- `/predict/polynomial` - Polynomial Regression predictions
- `/predict/logistic` - Logistic Regression classifications
- `/predict/knn` - KNN classifications 