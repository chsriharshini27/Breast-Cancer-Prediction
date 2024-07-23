# Breast-Cancer-Prediction

Data Preprocessing

Loading the data: Use pandas to load the dataset.
Exploratory Data Analysis (EDA): Analyze the distribution and correlation of features.
Data Cleaning: Handle missing values, if any.
Feature Scaling: Standardize features to have zero mean and unit variance.

#code
# Dropping unnecessary columns
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Mapping target values to binary
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separating features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


Model Selection

Algorithms: Commonly used algorithms include Logistic Regression, Support Vector Machine (SVM), Random Forest, and k-Nearest Neighbors (k-NN).
Train-Test Split: Split the dataset into training and testing sets.

Model Training

Training: Train the selected algorithms on the training set.
Hyperparameter Tuning: Use techniques like Grid Search or Random Search for tuning.

#code
# Initializing models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'k-NN': KNeighborsClassifier()
}

# Training and evaluating models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}\n")

Model Evaluation

Metrics: Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC score.
Confusion Matrix: Visualize the confusion matrix to understand the performance.

Model Deployment

Save the Model: Use libraries like pickle or joblib to save the trained model.
API Development: Develop a REST API using Flask or FastAPI for model inference.

libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle

