"""
Enhanced Docker Lab1 - Breast Cancer Classification with Evaluation Pipeline
This script loads the Breast Cancer Wisconsin dataset, preprocesses the data,
trains a Random Forest classifier, evaluates the model, and saves artifacts.
"""

import os
import json
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
print("Loading Breast Cancer Wisconsin dataset...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class names: {cancer.target_names}")
print()

# Step 2: Data Splitting
# Read environment variables with defaults
test_size = float(os.getenv('TEST_SIZE', '0.2'))
random_state = int(os.getenv('RANDOM_STATE', '42'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print()

# Step 3: Data Preprocessing
print("Preprocessing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preprocessing completed (StandardScaler)")
print()

# Step 4: Model Training
# Read hyperparameters from environment variables
n_estimators = int(os.getenv('N_ESTIMATORS', '100'))
max_depth_str = os.getenv('MAX_DEPTH', '10')
max_depth = int(max_depth_str) if max_depth_str.lower() != 'none' else None
model_random_state = int(os.getenv('RANDOM_STATE', '42'))

print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}...")
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=model_random_state
)

model.fit(X_train_scaled, y_train)
print("Model training completed")
print()

# Step 5: Model Evaluation
print("=== Model Evaluation ===")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print()

# Step 6: Save Artifacts
# Save model to models directory (created by Dockerfile)
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)  # Ensure directory exists

model_path = os.path.join(models_dir, 'cancer_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(models_dir, 'cancer_scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved: {scaler_path}")

# Create results dictionary
confusion_matrix_list = confusion_matrix(y_test, y_pred).tolist()
results = {
    'accuracy': float(accuracy),
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'test_size': test_size,
    'random_state': random_state,
    'confusion_matrix': confusion_matrix_list,
    'class_names': cancer.target_names.tolist()
}

# Save results to JSON
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved: evaluation_results.json")
print()
print("Model and evaluation results saved successfully!")
print(f"- Model: {model_path}")
print(f"- Scaler: {scaler_path}")
print("- Results: evaluation_results.json")

