from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sqlite3
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained XGBoost model
xgb_model = joblib.load('xgb_model.pkl')

# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Connect to SQLite database and retrieve data
    conn = sqlite3.connect('/Users/diboshbaruah/Desktop/Database.db')
    data = pd.read_sql_query('SELECT * FROM Heart_disease', conn)
    conn.close()

    # Data Pre-processing (similar to predict_model.py)
    categorical_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 
                           'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']

    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    bool_columns = data_encoded.select_dtypes(include='bool').columns
    data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

    data_encoded['HeartDisease'] = data_encoded['HeartDisease'].map({'Yes': 1, 'No': 0})

    numeric_columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    data_encoded[numeric_columns] = data_encoded[numeric_columns].apply(pd.to_numeric, errors='coerce')

    imputer = SimpleImputer(strategy='median')
    data_encoded[numeric_columns] = imputer.fit_transform(data_encoded[numeric_columns])

    missing_values = data_encoded.isnull().sum()
    if missing_values[missing_values > 0].any():
        print("There are still missing values after imputation.")

    expected_columns = [
        'BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 
        'Smoking_No', 'Smoking_Yes', 'AlcoholDrinking_Yes', 'Stroke_Yes', 
        'DiffWalking_No', 'DiffWalking_Yes', 'Sex_Female', 'Sex_Male', 
        'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 
        'AgeCategory_40-44', 'AgeCategory_45-49', 'AgeCategory_50-54', 
        'AgeCategory_55-59', 'AgeCategory_60-64', 'AgeCategory_65-69', 
        'AgeCategory_70-74', 'AgeCategory_75-79', 'AgeCategory_80 or older', 
        'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White', 
        'Diabetic_No', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes', 
        'Diabetic_Yes (during pregnancy)', 'PhysicalActivity_Yes', 'GenHealth_Excellent', 
        'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good', 
        'Asthma_No', 'Asthma_Yes', 'KidneyDisease_Yes', 'SkinCancer_No', 'SkinCancer_Yes'
    ]

    for col in expected_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    data_encoded = data_encoded[expected_columns]

    # Making predictions using the trained XGBoost model
    y_pred = xgb_model.predict(data_encoded)
    y_pred_proba = xgb_model.predict_proba(data_encoded)

    # Return prediction results as JSON
    prediction_result = {
        "prediction": 'Yes' if y_pred[0] == 1 else 'No',
        "probability_heart_disease": float(y_pred_proba[0][1]),
        "probability_no_heart_disease": float(y_pred_proba[0][0])
    }
    return jsonify(prediction_result)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
