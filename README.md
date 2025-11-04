# Enhanced Docker Lab1 - Breast Cancer Classification with Evaluation Pipeline

**Course**: IE7374 - MLOPs  
**Purpose**: Educational project demonstrating containerized machine learning model training and evaluation

## Project Overview

This project demonstrates a complete MLOps workflow for training and evaluating a machine learning model using Docker containers. The project trains a Random Forest classifier on the Breast Cancer Wisconsin dataset to classify tumors as malignant or benign. It includes data preprocessing, model evaluation, and artifact management following MLOps best practices.

### Key MLOps Concepts Demonstrated

1. **Containerization**: Using Docker to package ML training pipeline
2. **Reproducibility**: Environment variables for hyperparameter configuration
3. **Artifact Management**: Organized storage of trained models and preprocessing objects
4. **Model Evaluation**: Comprehensive metrics tracking and JSON-based results storage
5. **Version Control**: Docker images tagged for model versioning

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Type**: Binary classification
- **Samples**: 569 total instances
- **Features**: 30 numeric features derived from 10 core measurements:
  - Mean, standard error, and worst values for each feature
  - Features include: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **Target**: Binary classification
  - 0 = Malignant (cancerous)
  - 1 = Benign (non-cancerous)
- **Class Names**: `['malignant', 'benign']`

## Project Structure

```
.
├── dockerfile              # Docker configuration for containerized training
├── README.md               # This documentation file
└── src/
    ├── main.py            # Main training and evaluation script
    └── requirements.txt   # Python dependencies
```

### Generated Structure (After Running)

After executing the Docker container, the following structure will be created:

```
.
├── dockerfile
├── README.md
├── src/
│   ├── main.py
│   └── requirements.txt
├── models/                 # Created during execution
│   ├── cancer_model.pkl   # Trained Random Forest classifier
│   └── cancer_scaler.pkl  # Fitted StandardScaler for preprocessing
└── evaluation_results.json # Evaluation metrics and configuration
```

## Models Folder

The `models/` folder is automatically created during execution and contains:

### 1. `cancer_model.pkl`
- **Type**: Trained `RandomForestClassifier` model
- **Purpose**: This is the trained machine learning model that can make predictions
- **Usage**: Load using `joblib.load('models/cancer_model.pkl')` to make predictions on new data
- **Format**: Serialized using joblib (optimized for scikit-learn models)

### 2. `cancer_scaler.pkl`
- **Type**: Fitted `StandardScaler` object
- **Purpose**: Preprocessing transformer that standardizes features (mean=0, std=1)
- **Usage**: Must be applied to new data before making predictions
- **Importance**: Ensures new data is preprocessed identically to training data
- **Format**: Serialized using joblib


## Dependencies

All dependencies are listed in `src/requirements.txt`:

- **scikit-learn**: Machine learning library (datasets, models, preprocessing, metrics)
- **joblib**: Efficient serialization of scikit-learn models and numpy arrays
- **pandas**: Data manipulation and analysis (included for potential future enhancements)
- **numpy**: Numerical computing (required by scikit-learn)

## Environment Variables

The project uses environment variables for hyperparameter configuration, enabling easy experimentation and reproducibility. All variables have sensible defaults.

### Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `N_ESTIMATORS` | int | 100 | Number of trees in the Random Forest ensemble |
| `MAX_DEPTH` | int or None | 10 | Maximum depth of decision trees. Use `'none'` for unlimited depth |
| `TEST_SIZE` | float | 0.2 | Proportion of dataset reserved for testing (20%) |
| `RANDOM_STATE` | int | 42 | Random seed for reproducibility across data splitting and model training |

### Why Environment Variables?

- **Flexibility**: Change hyperparameters without modifying code
- **Reproducibility**: Same seed values produce identical results
- **Experimentation**: Easy A/B testing of different configurations
- **Containerization**: Docker best practice for configuration management

## Workflow

### Step-by-Step Process

1. **Data Loading**: Loads Breast Cancer Wisconsin dataset from sklearn
2. **Data Splitting**: Splits data into training and testing sets using configurable test size
3. **Data Preprocessing**: Applies StandardScaler to normalize features (critical for consistent model performance)
4. **Model Training**: Trains Random Forest classifier with configurable hyperparameters
5. **Model Evaluation**: Calculates accuracy, classification report, and confusion matrix
6. **Artifact Saving**: Saves model, scaler, and evaluation results

### Data Preprocessing

**StandardScaler** is used to:
- Normalize features to have zero mean and unit variance
- Improve model performance and convergence
- Ensure consistent scaling between training and inference

**Important**: The scaler is fitted only on training data to prevent data leakage, then applied to test data.

## Docker Commands

### Build the Docker Image

```bash
docker build -t breast-cancer-classifier:v1 .
```

**What this does**:
- Reads `dockerfile` to create a containerized environment
- Installs all Python dependencies from `requirements.txt`
- Creates the `models/` directory structure
- Sets default environment variables
- Prepares the image for execution

### Run with Default Parameters

```bash
docker run breast-cancer-classifier:v1
```

**Output**: Container runs the training script and generates all artifacts. Outputs are only visible in console (not persisted to host).

### Run with Custom Hyperparameters

```bash
docker run -e N_ESTIMATORS=200 -e MAX_DEPTH=15 breast-cancer-classifier:v1
```

**Use Cases**:
- Experimenting with different model configurations
- Tuning hyperparameters for better performance
- Testing model behavior with different parameters

### Run and Persist Outputs to Host

```bash
mkdir -p output
docker run -v $(pwd)/output:/app breast-cancer-classifier:v1
```

**What this does**:
- Creates `output/` directory on host machine
- Maps container's `/app` directory to host's `output/` directory
- All generated files (models, JSON results) are saved to host machine

**After execution, check**:
```bash
ls output/
# You'll see: models/ and evaluation_results.json
ls output/models/
# You'll see: cancer_model.pkl and cancer_scaler.pkl
```

### Run with Custom Parameters and Output Persistence

```bash
docker run -e N_ESTIMATORS=150 -e MAX_DEPTH=12 -v $(pwd)/output:/app breast-cancer-classifier:v1
```

**Combines**: Custom hyperparameters + output persistence for complete experimentation workflow.

## Output Files

### 1. `models/cancer_model.pkl`
Trained Random Forest classifier saved using joblib. This model can be loaded and used for predictions:

```python
import joblib
model = joblib.load('models/cancer_model.pkl')
predictions = model.predict(new_data)
```

### 2. `models/cancer_scaler.pkl`
Fitted StandardScaler that was used during training. **Must be applied** to new data before predictions:

```python
import joblib
scaler = joblib.load('models/cancer_scaler.pkl')
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
```

### 3. `evaluation_results.json`
JSON file containing comprehensive evaluation metrics and training configuration:

```json
{
  "accuracy": 0.9649,
  "n_estimators": 100,
  "max_depth": 10,
  "test_size": 0.2,
  "random_state": 42,
  "confusion_matrix": [[41, 2], [2, 69]],
  "class_names": ["malignant", "benign"]
}
```

**Fields Explained**:
- `accuracy`: Model accuracy on test set (0-1 scale)
- `n_estimators`: Number of trees used in Random Forest
- `max_depth`: Maximum tree depth used
- `test_size`: Proportion of data used for testing
- `random_state`: Random seed for reproducibility
- `confusion_matrix`: 2x2 matrix showing true vs predicted classifications
- `class_names`: List of class labels

## Expected Console Output

```
Loading Breast Cancer Wisconsin dataset...
Dataset shape: (569, 30)
Number of classes: 2
Class names: ['malignant' 'benign']

Training set size: 455
Test set size: 114

Preprocessing data...
Data preprocessing completed (StandardScaler)

Training Random Forest with n_estimators=100, max_depth=10...
Model training completed

=== Model Evaluation ===
Accuracy: 0.9649

Classification Report:
              precision    recall  f1-score   support

   malignant       0.96      0.95      0.96        43
      benign       0.97      0.98      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.96      0.96      0.96       114

Model saved: models/cancer_model.pkl
Scaler saved: models/cancer_scaler.pkl
Results saved: evaluation_results.json

Model and evaluation results saved successfully!
- Model: models/cancer_model.pkl
- Scaler: models/cancer_scaler.pkl
- Results: evaluation_results.json
```

## Evaluation Metrics Explained

### Accuracy
- **Definition**: Proportion of correct predictions
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 0.9649 = 96.49% of test samples correctly classified

### Classification Report
Provides detailed metrics for each class:

- **Precision**: Of all predictions for a class, how many were correct?
  - `TP / (TP + FP)`
- **Recall**: Of all actual instances of a class, how many did we find?
  - `TP / (TP + FN)`
- **F1-Score**: Harmonic mean of precision and recall
  - `2 * (precision * recall) / (precision + recall)`
- **Support**: Number of actual instances of each class in test set

### Confusion Matrix
2x2 matrix showing:
- **True Negatives (TN)**: Correctly predicted benign as benign
- **False Positives (FP)**: Incorrectly predicted benign as malignant
- **False Negatives (FN)**: Incorrectly predicted malignant as benign
- **True Positives (TP)**: Correctly predicted malignant as malignant

## MLOps Best Practices Demonstrated

1. ✅ **Containerization**: Entire training pipeline in Docker
2. ✅ **Reproducibility**: Fixed random seeds and versioned dependencies
3. ✅ **Configuration Management**: Environment variables for hyperparameters
4. ✅ **Artifact Management**: Organized model storage in dedicated folder
5. ✅ **Evaluation Tracking**: JSON-based results for logging and comparison
6. ✅ **Preprocessing Persistence**: Saved scaler ensures consistent data transformation
7. ✅ **Separation of Concerns**: Models separated from source code

## Use Cases

### For Development
- Experiment with different hyperparameters
- Compare model performance across configurations
- Understand ML pipeline workflow

### For Production (Future Enhancements)
- Models can be loaded by serving containers
- Scalers ensure consistent preprocessing in production
- JSON results can be logged to experiment tracking systems

## Troubleshooting

### Issue: Models folder not created
**Solution**: Ensure Docker has write permissions. The Dockerfile creates the folder, but verify with `docker run` commands.

### Issue: Different results on rerun
**Solution**: Ensure `RANDOM_STATE` environment variable is set consistently.

### Issue: Cannot load model in another script
**Solution**: Ensure you load both the scaler and model, and apply scaler to new data before predictions.

## Next Steps (Lab2)

This Lab1 focuses on training and evaluation. Lab2 will add:
- Multi-stage Docker builds
- Web server (Flask) for model serving
- REST API endpoints
- HTML templates for user interface

## License & Course Information

**Course**: IE7374 - Machine Learning Operations (MLOps)  
**Purpose**: Educational project for learning containerized ML workflows  
**Dataset**: Public dataset from scikit-learn (Breast Cancer Wisconsin)

---

**Note**: This project is designed for educational purposes to understand MLOps principles including containerization, artifact management, and reproducible ML workflows.
