import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv("experience_income_dataset.csv")

# Rename columns for consistency
df.rename(columns={"Experience (Years)": "YearsExperience", "Monthly Income": "Income"}, inplace=True)

# ❌ Do NOT convert Monthly Income to Annual Income
# df["Income"] = df["Income"] * 12  # Ensure this is commented out

# Define independent (X) and dependent (Y) variables
X = df[["YearsExperience"]]
Y = df["Income"]

# Apply Polynomial Regression (Degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_poly, Y)

# Save both poly and model correctly
with open("model.pkl", "wb") as file:
    pickle.dump((poly, model), file)  # ✅ Save both transformer and model

print("✅ Model trained and saved as model.pkl with Monthly Income!\n")

# Function to predict income
def predict_income(experience):
    # Convert to DataFrame to avoid warning
    exp_input = pd.DataFrame([[experience]], columns=["YearsExperience"])
    predicted_income = model.predict(poly.transform(exp_input))[0]
    return predicted_income

# Test Predictions
years = [5, 10, 25, 60]
for exp in years:
    predicted_income = predict_income(exp)
    print(f"Predicted Income for {exp} years of experience: ${predicted_income:,.2f}")