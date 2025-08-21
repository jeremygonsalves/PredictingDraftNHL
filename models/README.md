# Models Directory

This directory contains the machine learning models and related utilities for the NHL Draft Prediction project.

## Directory Structure

```
models/
├── saved/                  # Saved trained models
├── README.md              # This file
└── src/models/            # Model source code
    ├── model.py           # Custom model implementations
    ├── setup_predictor.py # Model pipeline setup
    └── train_test_predictor.py # Training and evaluation utilities
```

## Model Architecture

### Custom Models

The project implements several custom ordinal classification models:

#### 1. LogisticOrdinalRegression
- **Type**: Ordinal logistic regression
- **Use Case**: Baseline model for draft position prediction
- **Features**: Handles ordinal nature of draft positions

#### 2. RandomForestOrdinalClassifier
- **Type**: Random Forest adapted for ordinal classification
- **Use Case**: Ensemble method for draft prediction
- **Features**: Handles non-linear relationships

#### 3. SVMOrdinalClassifier
- **Type**: Support Vector Machine for ordinal classification
- **Use Case**: High-dimensional feature space
- **Features**: Kernel-based learning

#### 4. MLPOrdinalClassifier
- **Type**: Multi-Layer Perceptron for ordinal classification
- **Use Case**: Deep learning approach
- **Features**: Neural network architecture

### Pipeline Setup

The `setup_predictor.py` module provides a unified interface for creating machine learning pipelines:

```python
from src.models.setup_predictor import setup
from src.models.model import RandomForestOrdinalClassifier

# Create pipeline
pipeline = setup(
    numeric_cols=['Height', 'Weight'],
    categorical_cols=['Position'],
    text_cols=['scouting_report'],
    func=RandomForestOrdinalClassifier(),
    use_bert=True
)
```

### Feature Engineering

The models use a combination of:

1. **Numeric Features**:
   - Height and weight
   - BERT embeddings (768 dimensions)

2. **Categorical Features**:
   - Player position (one-hot encoded)
   - Reporter source (one-hot encoded)

3. **Text Features**:
   - Scouting reports (BERT or TF-IDF embeddings)

## Model Training

### Training Process

1. **Data Preprocessing**:
   - Handle missing values
   - Scale numeric features
   - Encode categorical variables
   - Generate text embeddings

2. **Model Training**:
   - Cross-validation (5-fold)
   - Hyperparameter tuning
   - Model selection

3. **Evaluation**:
   - Accuracy, F1-score, Precision, Recall
   - Cross-validation scores
   - Model comparison

### Usage Example

```python
from src.models.train_test_predictor import train_and_test
from src.models.setup_predictor import setup
from src.models.model import RandomForestOrdinalClassifier

# Setup model
model = setup(
    numeric_cols=numeric_features,
    categorical_cols=categorical_features,
    text_cols=text_features,
    func=RandomForestOrdinalClassifier()
)

# Train and evaluate
results = train_and_test(model, X, y, groups=player_names)
```

## Model Performance

### Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Cross-Validation

- **Method**: 5-fold stratified cross-validation
- **Stratification**: Based on draft position ranges
- **Scoring**: Accuracy and F1-score

## Model Persistence

Trained models can be saved and loaded:

```python
import joblib

# Save model
joblib.dump(pipeline, 'models/saved/best_model.pkl')

# Load model
loaded_model = joblib.load('models/saved/best_model.pkl')
```

## Hyperparameter Tuning

Each model supports hyperparameter optimization:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

## Model Interpretability

- **Feature Importance**: Available for tree-based models
- **SHAP Values**: For model explanation
- **Partial Dependence Plots**: For feature effects

## Best Practices

1. **Data Validation**: Ensure data quality before training
2. **Feature Selection**: Use domain knowledge to select relevant features
3. **Cross-Validation**: Always use cross-validation for reliable estimates
4. **Model Comparison**: Compare multiple models before final selection
5. **Hyperparameter Tuning**: Optimize hyperparameters for best performance 