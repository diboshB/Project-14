# Project-14
Capstone_Project - Classification - Heart_Disease_Prediction


1. Introduction
Heart disease prediction is a critical healthcare problem, where the objective
is to predict the likelihood of a patient suffering from heart disease based on
various health metrics. Early and accurate detection can help mitigate the
risk and provide timely interventions. For this project, we used a dataset
containing medical records of patients, with the target variable,
HeartDisease, indicating whether the patient has heart disease (1) or not
(0). The goal of this project was to build a machine learning model that can
predict heart disease with high accuracy using the XGBoost algorithm,
known for its high performance in classification tasks.
2. Data Collection
The dataset for this project contains medical records of individuals, with
several features related to their health conditions, lifestyle, and medical
history. The features include variables such as age, sex, blood pressure,
cholesterol levels, and more, which are important indicators of heart
disease. The target variable is HeartDisease, where 1 indicates the presence
of heart disease and 0 indicates its absence.
Once the data was collected, it was loaded into a Pandas DataFrame for
further preprocessing.
3. Data Preprocessing
Before applying the machine learning model, we performed several
preprocessing steps to clean and transform the data:
3.1 Handling Missing Values
The dataset had no missing values after checking with isnull(). Thus, no
imputation was necessary, and the data was ready for processing.
3.2 Feature Engineering
We used the dataset's features directly without additional transformations,
as the variables were already meaningful for the task. However, categorical
variables, such as sex, were encoded using one-hot encoding if necessary.
3.3 Feature Scaling
While XGBoost can handle unscaled features, we performed scaling on
continuous variables (e.g., age, cholesterol, and blood pressure) to ensure
that no feature disproportionately influenced the model due to its scale.
4. Model Building
For this predictive task, we chose XGBoost, a gradient boosting machine
learning algorithm that has shown strong performance in various
classification problems. The steps involved in building the model are outlined
below:
4.1 Splitting the Dataset
The dataset was split into features (X) and the target variable (y). Then, it
was further divided into training and testing sets, with 80% of the data used
for training and 20% for testing:
4.2 Training the XGBoost Model
An XGBoost classifier was initialized with 100 estimators, and the model was
trained using the training data:
4.3 Making Predictions
After training, the model was used to make predictions on the test set:
5. Model Evaluation
The model’s performance was evaluated using several metrics to understand
its effectiveness:
• Accuracy: The overall percentage of correctly classified instances.
• Precision: The proportion of true positives among the instances
classified as positive.
• Recall: The proportion of true positives among all actual positive
instances.
• F1-Score: The harmonic mean of precision and recall.
• AUC-ROC Curve: The Area Under the Receiver Operating
Characteristic Curve, which shows the model's ability to distinguish
between the two classes.
5.1 Performance Metrics
Results:
• Accuracy: 91.38%
• Classification Report:
o Precision:
§ Class 0 (No Heart Disease): 0.92
§ Class 1 (Heart Disease): 0.54
o Recall:
§ Class 0 (No Heart Disease): 0.99
§ Class 1 (Heart Disease): 0.10
o F1-Score: 0.56 (macro avg)
o Weighted avg: 0.89
• Confusion Matrix:
o True Negatives (TN): 57,906
o False Positives (FP): 461
o False Negatives (FN): 5,053
o True Positives (TP): 539
6. Results & Discussion
6.1 Model Performance
The XGBoost model performed well overall, achieving an accuracy of
91.38%. However, the recall for heart disease patients (Class 1) was quite
low at 10%, indicating that many patients with heart disease were not
detected by the model. This may be a result of the class imbalance, as the
dataset has far more instances of non-heart disease than heart disease.
• Precision for class 1 (Heart Disease) is 0.54, meaning that half of the
predicted heart disease cases were correct.
• Recall for class 1 is low at 0.10, highlighting a significant issue with
detecting heart disease cases.
This suggests that the model is biased towards predicting the majority class
(non-heart disease).
6.2 Feature Importance
We also examined the feature importance scores from XGBoost to
understand which variables had the most influence on the model's decisions.
The top features included:
• Age
• Chest pain type
• Blood pressure
• Cholesterol levels
• Maximum heart rate achieved
These features were deemed the most influential in predicting heart disease,
and their importance could guide future improvements in model performance
or feature selection.
7. Conclusion
This project demonstrates the use of XGBoost for predicting heart disease
based on medical features. The model achieved a high accuracy but showed
room for improvement in detecting heart disease cases (low recall for class
1). To address the class imbalance, techniques like oversampling (e.g.,
SMOTE) or adjusting class weights could be explored to improve recall for
the minority class. Future work could focus on experimenting with other
models or ensemble techniques to enhance predictive performance further.
Deploying such a model in clinical settings could assist healthcare
professionals in identifying high-risk patients and prioritizing interventions,
ultimately reducing heart disease-related mortality rates.
