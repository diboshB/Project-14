# train_model.py

import sqlite3 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib 


# Connecting to database.db and reading data
conn = sqlite3.connect('/Users/diboshbaruah/Desktop/Database.db')
data = pd.read_sql_query('SELECT * FROM Heart_disease', conn)

# Closing the connection
conn.close()

# Data Pre-processing

# Identifying categorical columns
categorical_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 
                       'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']

# Creating dummy variables for categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Converting all boolean columns to integers (0 and 1)
bool_columns = data_encoded.select_dtypes(include='bool').columns
data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

# Converting 'HeartDisease' from 'Yes'/'No' to 1/0
data_encoded['HeartDisease'] = data_encoded['HeartDisease'].map({'Yes': 1, 'No': 0})

# Converting 'BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime' to numeric
numeric_columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
data_encoded[numeric_columns] = data_encoded[numeric_columns].apply(pd.to_numeric, errors='coerce')

print("Data pre-processing completed successfully.")

# Impute missing values (using median strategy)
imputer = SimpleImputer(strategy='median')
data_encoded[numeric_columns] = imputer.fit_transform(data_encoded[numeric_columns])

# Separating features and target variable
X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the resampled data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print()
print("Model training started using XGBOOST...")
print()

# Initialize the XGBoost classifier with tuned parameters
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42, eval_metric='mlogloss')

# Training the model on the training data
xgb_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print()

# Saving the trained model
joblib.dump(xgb_model, 'xgb_model.pkl')
print("Model saved as 'xgb_model.pkl'")
