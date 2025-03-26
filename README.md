# Experience-Income Prediction Models

This web application provides multiple regression models for predicting income based on various features. The application includes:

- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- Polynomial Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)

## Features

- Interactive web interface for model selection
- Real-time predictions
- Support for multiple input features
- Classification and regression capabilities
- Modern and responsive UI

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset file (`data.csv`) in the project root directory. The dataset should contain the following columns:
   - Experience
   - EducationLevel
   - Certifications
   - JobRole
   - MonthlyIncome

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to `http://localhost:5000`

## Model Details

### Simple Linear Regression
- Predicts income based on years of experience
- Uses a single feature for prediction

### Multiple Linear Regression
- Predicts income using multiple features:
  - Experience
  - Education Level
  - Certifications
  - Job Role

### Polynomial Regression
- Handles non-linear relationships
- Uses polynomial features of experience

### Logistic Regression
- Classifies income as High or Low
- Uses multiple features for classification
- Based on median income threshold

### K-Nearest Neighbors
- Classifies income level using KNN algorithm
- Uses experience and education level
- Based on median income threshold

## Usage

1. Select a model from the main page
2. Enter the required input features
3. Click the predict/classify button
4. View the prediction result

## Data Format

The input dataset should be in CSV format with the following columns:
- Experience: Years of experience (numeric)
- EducationLevel: Education level (1-5)
- Certifications: Number of certifications (numeric)
- JobRole: Job role level (1-5)
- MonthlyIncome: Monthly income (numeric)

## Note

Make sure your dataset is properly formatted and contains all the required columns before running the application. 