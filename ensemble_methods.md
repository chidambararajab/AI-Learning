# Ensemble Methods in Machine Learning: Random Forests and Gradient Boosting

## Introduction to Ensemble Methods

Ensemble methods are machine learning techniques that combine multiple individual models (called base learners or weak learners) to create a more powerful predictive model. The fundamental idea behind ensembles is that a group of weak learners can come together to form a strong learner, often outperforming any single model.

The effectiveness of ensemble methods comes from their ability to:
- Reduce variance (by averaging multiple models)
- Reduce bias (by combining different types of models)
- Avoid overfitting (through various regularization techniques)
- Improve overall predictive performance

### Types of Ensemble Methods

There are three main categories of ensemble methods:

1. **Bagging (Bootstrap Aggregating)**: Creates multiple versions of the training dataset through bootstrap sampling (random sampling with replacement) and trains a model on each sample. Predictions are combined through averaging (for regression) or voting (for classification).
   - Example: Random Forests

2. **Boosting**: Trains models sequentially, where each new model focuses on the errors made by previous models. Combines models by weighted majority vote or weighted sum.
   - Examples: AdaBoost, Gradient Boosting Machines (GBM), XGBoost, LightGBM, CatBoost

3. **Stacking**: Trains multiple different models and combines their predictions using another model (meta-learner) that learns how to best combine the predictions.
   - Example: Stacked Generalization

This document will focus primarily on two of the most powerful and widely used ensemble methods: Random Forests (a bagging method) and Gradient Boosting (a boosting method).

## Random Forests

### Theoretical Foundation

Random Forest is an ensemble of decision trees, typically trained via the bagging method. It was developed by Leo Breiman in 2001 and has become one of the most popular machine learning algorithms due to its effectiveness and simplicity.

#### How Random Forests Work:

1. **Bootstrap Sampling**: Randomly sample the training data with replacement to create multiple datasets.
2. **Random Feature Selection**: When building each tree, only consider a random subset of features at each split.
3. **Decision Tree Building**: Grow a decision tree on each bootstrapped dataset with the random feature selection constraint.
4. **Aggregation**: Combine predictions from all trees by majority vote (classification) or averaging (regression).

The two sources of randomness (bootstrap sampling and feature selection) ensure that the trees in the forest are decorrelated, which is key to the algorithm's success.

#### Key Hyperparameters:

- **n_estimators**: The number of trees in the forest
- **max_depth**: Maximum depth of each tree
- **max_features**: Maximum number of features to consider for each split
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required in a leaf node
- **bootstrap**: Whether to use bootstrap sampling

### Advantages of Random Forests:

1. **Performance**: Often provides high accuracy without extensive hyperparameter tuning
2. **Robustness**: Resistant to overfitting, especially with a large number of trees
3. **Feature Importance**: Built-in ability to rank feature importance
4. **Handles Non-linearity**: Effectively captures non-linear relationships and interactions
5. **Missing Values**: Can handle missing values through proximities
6. **Parallelization**: Can be easily parallelized as trees are independent

### Limitations of Random Forests:

1. **Interpretability**: Less interpretable than a single decision tree
2. **Computational Resources**: Training many deep trees can be resource-intensive
3. **Bias Toward Categorical Features**: Biased toward features with more levels
4. **Not Optimal for Linear Relationships**: May be outperformed by linear models when the true relationship is linear

## Gradient Boosting

### Theoretical Foundation

Gradient Boosting is a sequential ensemble method that builds models iteratively, with each new model correcting the errors of its predecessors. It uses the gradient descent algorithm to minimize a loss function by adding weak learners (usually decision trees) that focus on the instances that previous models misclassified or where they made large errors.

#### How Gradient Boosting Works:

1. **Initialize Model**: Start with a simple model (often just a single prediction for all instances)
2. **Compute Residuals**: Calculate the residual errors (difference between actual and predicted values)
3. **Train Weak Learner**: Fit a weak learner (typically a shallow decision tree) to predict the residuals
4. **Update Model**: Add the new weak learner to the ensemble with a weight (learning rate)
5. **Iterate**: Repeat steps 2-4 for a specified number of iterations

Unlike Random Forests, where trees are built independently, Gradient Boosting builds trees sequentially, with each tree trying to correct the mistakes of the combined ensemble so far.

#### Key Hyperparameters:

- **n_estimators**: Number of boosting stages (trees)
- **learning_rate**: Contribution of each tree to the final solution (shrinkage)
- **max_depth**: Maximum depth of each tree
- **subsample**: Fraction of samples to use for fitting individual trees
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required in a leaf node
- **max_features**: Maximum number of features to consider for each split

### Popular Gradient Boosting Implementations:

1. **GBM (Gradient Boosting Machine)**: The classic implementation (in scikit-learn)
2. **XGBoost**: Optimized for performance and speed
3. **LightGBM**: Designed for efficiency with large datasets
4. **CatBoost**: Handles categorical features automatically and reduces overfitting

### Advantages of Gradient Boosting:

1. **Performance**: Often achieves state-of-the-art results on many datasets
2. **Flexibility**: Works well with different loss functions
3. **Feature Importance**: Provides measures of feature importance
4. **Handles Mixed Data**: Works well with both numerical and categorical features
5. **Customization**: Highly customizable for specific problems

### Limitations of Gradient Boosting:

1. **Sensitivity to Hyperparameters**: Requires careful tuning to avoid overfitting
2. **Training Time**: Sequential nature makes it harder to parallelize
3. **Memory Usage**: Can be memory-intensive, especially with deep trees
4. **Overfitting Risk**: More prone to overfitting than Random Forests, especially with too many trees or high learning rates

## Real-World Example: Credit Risk Assessment

Let's implement both Random Forests and Gradient Boosting models for a credit risk assessment task. We'll predict whether a loan applicant is likely to default based on various financial and personal attributes.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic credit risk data
def generate_credit_data(n_samples=10000):
    """Generate synthetic credit risk data."""
    data = {
        # Numerical Features
        'income': np.random.lognormal(mean=10.5, sigma=0.4, size=n_samples),
        'age': np.random.normal(loc=40, scale=10, size=n_samples),
        'loan_amount': np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60, 72, 84], size=n_samples),
        'interest_rate': np.random.uniform(low=5.0, high=25.0, size=n_samples),
        'credit_score': np.random.normal(loc=680, scale=80, size=n_samples),
        'debt_to_income': np.random.uniform(low=0.0, high=0.5, size=n_samples),
        'num_credit_lines': np.random.poisson(lam=5, size=n_samples),
        'num_late_payments': np.random.poisson(lam=0.5, size=n_samples),
        'months_employed': np.random.poisson(lam=50, size=n_samples),
        
        # Categorical Features
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'],
                                            p=[0.7, 0.15, 0.05, 0.1], size=n_samples),
        'housing_status': np.random.choice(['Own', 'Mortgage', 'Rent'], 
                                         p=[0.3, 0.4, 0.3], size=n_samples),
        'loan_purpose': np.random.choice(['Home', 'Auto', 'Education', 'Medical', 'Debt Consolidation', 'Other'],
                                       p=[0.2, 0.2, 0.1, 0.1, 0.3, 0.1], size=n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', 'Other'],
                                    p=[0.3, 0.4, 0.2, 0.05, 0.05], size=n_samples)
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Introduce some realistic correlations and effects
    df['credit_score'] = df['credit_score'] - 100 * (df['num_late_payments'] > 2).astype(int)
    df['interest_rate'] = df['interest_rate'] + 5 * (df['credit_score'] < 600).astype(int)
    
    # Generate target (default status)
    # Higher probability of default for certain conditions
    default_prob = 0.05  # Base default rate is 5%
    
    # Add weightings for different features
    default_prob += 0.15 * (df['credit_score'] < 600).astype(int)
    default_prob += 0.1 * (df['num_late_payments'] > 3).astype(int)
    default_prob += 0.08 * (df['debt_to_income'] > 0.4).astype(int)
    default_prob += 0.05 * (df['employment_status'] == 'Unemployed').astype(int)
    default_prob += 0.05 * (df['income'] < 30000).astype(int)
    default_prob -= 0.03 * (df['credit_score'] > 750).astype(int)
    
    # Ensure probabilities are between 0 and 1
    default_prob = np.clip(default_prob, 0, 1)
    
    # Generate default status based on probabilities
    df['default'] = np.random.binomial(n=1, p=default_prob)
    
    # Introduce some missing values
    for col in ['income', 'credit_score', 'months_employed', 'education']:
        mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    return df

# Generate the data
credit_data = generate_credit_data(n_samples=10000)

# Display basic information about the dataset
print("Dataset shape:", credit_data.shape)
print("\nDefault rate:", credit_data['default'].mean())

print("\nSample of the dataset:")
print(credit_data.head())

print("\nData types:")
print(credit_data.dtypes)

print("\nMissing values:")
print(credit_data.isnull().sum())

# Basic Exploratory Data Analysis (EDA)
print("\nStatistical summary of numerical features:")
print(credit_data.describe())

# Visualize default rate
plt.figure(figsize=(6, 4))
sns.countplot(x='default', data=credit_data)
plt.title('Distribution of Default Status')
plt.tight_layout()
plt.savefig('default_distribution.png')

# Correlation matrix of numerical features
numeric_features = credit_data.select_dtypes(include=['int64', 'float64']).columns
correlation = credit_data[numeric_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Split the data
X = credit_data.drop('default', axis=1)
y = credit_data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Define preprocessing for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 1. Random Forest Implementation

# Define the Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid for Random Forest
rf_param_grid = {
    'classifier__n_estimators': [100, 200],  # Number of trees
    'classifier__max_depth': [None, 10, 20],  # Maximum depth of trees
    'classifier__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'classifier__max_features': ['sqrt', 'log2']  # Number of features to consider for best split
}

# Create grid search with cross-validation
rf_cv = GridSearchCV(
    rf_pipeline, 
    rf_param_grid, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Train the Random Forest with grid search
print("\nTraining Random Forest with Grid Search...")
rf_cv.fit(X_train, y_train)

print("\nBest Random Forest parameters:")
print(rf_cv.best_params_)

# Get the best model
rf_best = rf_cv.best_estimator_

# Make predictions
rf_train_pred = rf_best.predict(X_train)
rf_test_pred = rf_best.predict(X_test)
rf_test_prob = rf_best.predict_proba(X_test)[:, 1]

# Evaluate Random Forest performance
rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
rf_test_precision = precision_score(y_test, rf_test_pred)
rf_test_recall = recall_score(y_test, rf_test_pred)
rf_test_f1 = f1_score(y_test, rf_test_pred)
rf_test_auc = roc_auc_score(y_test, rf_test_prob)

print("\nRandom Forest Performance:")
print(f"Training Accuracy: {rf_train_accuracy:.4f}")
print(f"Test Accuracy: {rf_test_accuracy:.4f}")
print(f"Test Precision: {rf_test_precision:.4f}")
print(f"Test Recall: {rf_test_recall:.4f}")
print(f"Test F1 Score: {rf_test_f1:.4f}")
print(f"Test AUC-ROC: {rf_test_auc:.4f}")

# Display classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_test_pred))

# Plot confusion matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png')

# 2. Gradient Boosting Implementation

# Define the Gradient Boosting pipeline
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define hyperparameter grid for Gradient Boosting
gb_param_grid = {
    'classifier__n_estimators': [100, 200],  # Number of boosting stages
    'classifier__learning_rate': [0.01, 0.1],  # Learning rate
    'classifier__max_depth': [3, 5, 7],  # Maximum depth of individual trees
    'classifier__min_samples_split': [2, 5],  # Minimum samples required to split a node
    'classifier__subsample': [0.8, 1.0]  # Fraction of samples for fitting trees
}

# Create grid search with cross-validation
gb_cv = GridSearchCV(
    gb_pipeline, 
    gb_param_grid, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Train the Gradient Boosting with grid search
print("\nTraining Gradient Boosting with Grid Search...")
gb_cv.fit(X_train, y_train)

print("\nBest Gradient Boosting parameters:")
print(gb_cv.best_params_)

# Get the best model
gb_best = gb_cv.best_estimator_

# Make predictions
gb_train_pred = gb_best.predict(X_train)
gb_test_pred = gb_best.predict(X_test)
gb_test_prob = gb_best.predict_proba(X_test)[:, 1]

# Evaluate Gradient Boosting performance
gb_train_accuracy = accuracy_score(y_train, gb_train_pred)
gb_test_accuracy = accuracy_score(y_test, gb_test_pred)
gb_test_precision = precision_score(y_test, gb_test_pred)
gb_test_recall = recall_score(y_test, gb_test_pred)
gb_test_f1 = f1_score(y_test, gb_test_pred)
gb_test_auc = roc_auc_score(y_test, gb_test_prob)

print("\nGradient Boosting Performance:")
print(f"Training Accuracy: {gb_train_accuracy:.4f}")
print(f"Test Accuracy: {gb_test_accuracy:.4f}")
print(f"Test Precision: {gb_test_precision:.4f}")
print(f"Test Recall: {gb_test_recall:.4f}")
print(f"Test F1 Score: {gb_test_f1:.4f}")
print(f"Test AUC-ROC: {gb_test_auc:.4f}")

# Display classification report
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_test_pred))

# Plot confusion matrix for Gradient Boosting
gb_cm = confusion_matrix(y_test, gb_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Default', 'Default'],
            yticklabels=['Not Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Gradient Boosting Confusion Matrix')
plt.tight_layout()
plt.savefig('gb_confusion_matrix.png')

# 3. XGBoost Implementation

# Define the XGBoost pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Define hyperparameter grid for XGBoost
xgb_param_grid = {
    'classifier__n_estimators': [100, 200],  # Number of boosting rounds
    'classifier__learning_rate': [0.01, 0.1],  # Learning rate
    'classifier__max_depth': [3, 5, 7],  # Maximum tree depth
    'classifier__subsample': [0.8, 1.0],  # Subsample ratio of the training instances
    'classifier__colsample_bytree': [0.8, 1.0]  # Subsample ratio of columns when constructing each tree
}

# Create grid search with cross-validation
xgb_cv = GridSearchCV(
    xgb_pipeline, 
    xgb_param_grid, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Train the XGBoost with grid search
print("\nTraining XGBoost with Grid Search...")
xgb_cv.fit(X_train, y_train)

print("\nBest XGBoost parameters:")
print(xgb_cv.best_params_)

# Get the best model
xgb_best = xgb_cv.best_estimator_

# Make predictions
xgb_train_pred = xgb_best.predict(X_train)
xgb_test_pred = xgb_best.predict(X_test)
xgb_test_prob = xgb_best.predict_proba(X_test)[:, 1]

# Evaluate XGBoost performance
xgb_train_accuracy = accuracy_score(y_train, xgb_train_pred)
xgb_test_accuracy = accuracy_score(y_test, xgb_test_pred)
xgb_test_precision = precision_score(y_test, xgb_test_pred)
xgb_test_recall = recall_score(y_test, xgb_test_pred)
xgb_test_f1 = f1_score(y_test, xgb_test_pred)
xgb_test_auc = roc_auc_score(y_test, xgb_test_prob)

print("\nXGBoost Performance:")
print(f"Training Accuracy: {xgb_train_accuracy:.4f}")
print(f"Test Accuracy: {xgb_test_accuracy:.4f}")
print(f"Test Precision: {xgb_test_precision:.4f}")
print(f"Test Recall: {xgb_test_recall:.4f}")
print(f"Test F1 Score: {xgb_test_f1:.4f}")
print(f"Test AUC-ROC: {xgb_test_auc:.4f}")

# Display classification report
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_test_pred))

# 4. Compare Model Performance

# Create a comparison table
models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
accuracy = [rf_test_accuracy, gb_test_accuracy, xgb_test_accuracy]
precision = [rf_test_precision, gb_test_precision, xgb_test_precision]
recall = [rf_test_recall, gb_test_recall, xgb_test_recall]
f1 = [rf_test_f1, gb_test_f1, xgb_test_f1]
auc = [rf_test_auc, gb_test_auc, xgb_test_auc]

comparison = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'AUC-ROC': auc
})

print("\nModel Comparison:")
print(comparison)

# Plot ROC curves for comparison
plt.figure(figsize=(10, 8))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_prob)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_test_auc:.3f})')

# Gradient Boosting ROC
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_test_prob)
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {gb_test_auc:.3f})')

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_test_prob)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_test_auc:.3f})')

# Plot details
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png')

# 5. Feature Importance Analysis

# Get feature names after preprocessing
def get_feature_names(column_transformer):
    """Get feature names from column transformer."""
    output_features = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(pipe, 'get_feature_names_out'):
                # For new scikit-learn versions
                current_features = pipe.get_feature_names_out()
            elif hasattr(pipe.steps[-1][1], 'get_feature_names_out'):
                # For pipeline transformers
                current_features = pipe.steps[-1][1].get_feature_names_out()
            else:
                # For older scikit-learn versions or if not available
                current_features = [f"{name}_{f}" for f in features]
                
            output_features.extend(current_features)
                
    return output_features

# Try to get feature names (this is approximate and may not be perfectly accurate)
try:
    preprocessor_fitted = rf_best.named_steps['preprocessor']
    feature_names = get_feature_names(preprocessor_fitted)
except:
    # Fallback if getting feature names fails
    feature_names = [f"feature_{i}" for i in range(100)]  # Arbitrary number of features

# Extract feature importances from Random Forest
rf_importances = rf_best.named_steps['classifier'].feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': feature_names[:len(rf_importances)],
    'Importance': rf_importances
}).sort_values('Importance', ascending=False).head(15)

# Extract feature importances from Gradient Boosting
gb_importances = gb_best.named_steps['classifier'].feature_importances_
gb_importance_df = pd.DataFrame({
    'Feature': feature_names[:len(gb_importances)],
    'Importance': gb_importances
}).sort_values('Importance', ascending=False).head(15)

# Extract feature importances from XGBoost
xgb_importances = xgb_best.named_steps['classifier'].feature_importances_
xgb_importance_df = pd.DataFrame({
    'Feature': feature_names[:len(xgb_importances)],
    'Importance': xgb_importances
}).sort_values('Importance', ascending=False).head(15)

# Plot feature importances
plt.figure(figsize=(20, 15))

# Random Forest feature importance
plt.subplot(3, 1, 1)
sns.barplot(x='Importance', y='Feature', data=rf_importance_df)
plt.title('Random Forest Feature Importance')

# Gradient Boosting feature importance
plt.subplot(3, 1, 2)
sns.barplot(x='Importance', y='Feature', data=gb_importance_df)
plt.title('Gradient Boosting Feature Importance')

# XGBoost feature importance
plt.subplot(3, 1, 3)
sns.barplot(x='Importance', y='Feature', data=xgb_importance_df)
plt.title('XGBoost Feature Importance')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png')

print("\nTop 10 Important Features (Random Forest):")
print(rf_importance_df.head(10))

# 6. Model Interpretation with SHAP (SHapley Additive exPlanations)
# This is an advanced technique for interpreting model predictions

# Convert test data for SHAP
X_test_processed = rf_best.named_steps['preprocessor'].transform(X_test)

# Create a sample for SHAP analysis (using a few test instances for clarity)
shap_sample = X_test_processed[:100]  # Use 100 samples for demonstration

# SHAP for Random Forest
try:
    print("\nCalculating SHAP values for Random Forest...")
    rf_explainer = shap.TreeExplainer(rf_best.named_steps['classifier'])
    rf_shap_values = rf_explainer.shap_values(shap_sample)
    
    # SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(rf_shap_values[1], shap_sample, feature_names=feature_names[:shap_sample.shape[1]], show=False)
    plt.title('SHAP Summary Plot for Random Forest (Default Class)')
    plt.tight_layout()
    plt.savefig('rf_shap_summary.png')
    plt.close()
    
    # SHAP waterfall plot for a single instance
    plt.figure(figsize=(12, 8))
    shap.plots._waterfall.waterfall_legacy(rf_explainer.expected_value[1], rf_shap_values[1][0], 
                                    feature_names=feature_names[:shap_sample.shape[1]], show=False)
    plt.title('SHAP Waterfall Plot for Single Instance (Random Forest)')
    plt.tight_layout()
    plt.savefig('rf_shap_waterfall.png')
    plt.close()
    
except Exception as e:
    print(f"SHAP visualization failed: {e}")
    print("Skipping SHAP analysis due to error.")

# 7. Predict for a new applicant using the best model

def predict_default_risk(applicant_data, model):
    """
    Predict the default risk for a loan applicant.
    
    Args:
        applicant_data: Dictionary with applicant features
        model: Trained pipeline model
        
    Returns:
        Prediction and probability
    """
    # Convert to DataFrame
    applicant_df = pd.DataFrame([applicant_data])
    
    # Make prediction
    default_prob = model.predict_proba(applicant_df)[0, 1]
    default_pred = model.predict(applicant_df)[0]
    
    # Define risk categories
    if default_prob < 0.1:
        risk_category = "Very Low Risk"
    elif default_prob < 0.25:
        risk_category = "Low Risk"
    elif default_prob < 0.5:
        risk_category = "Moderate Risk"
    elif default_prob < 0.75:
        risk_category = "High Risk"
    else:
        risk_category = "Very High Risk"
    
    return default_pred, default_prob, risk_category

# Example: New loan applicant
new_applicant = {
    'income': 65000,
    'age': 35,
    'loan_amount': 25000,
    'loan_term': 36,
    'interest_rate': 8.5,
    'credit_score': 710,
    'debt_to_income': 0.28,
    'num_credit_lines': 5,
    'num_late_payments': 0,
    'months_employed': 60,
    'employment_status': 'Employed',
    'housing_status': 'Mortgage',
    'loan_purpose': 'Auto',
    'education': 'Bachelor'
}

# Choose the best model based on AUC
best_model = max([rf_best, gb_best, xgb_best], key=lambda m: m.score(X_test, y_test))
best_model_name = {rf_best: 'Random Forest', gb_best: 'Gradient Boosting', xgb_best: 'XGBoost'}[best_model]

# Predict for the new applicant
default_prediction, default_probability, risk_category = predict_default_risk(new_applicant, best_model)

print(f"\nUsing {best_model_name} for prediction (best performing model)")
print("\nNew Applicant Prediction:")
print(f"Default Prediction: {'Will Default' if default_prediction == 1 else 'Will Not Default'}")
print(f"Default Probability: {default_probability:.4f}")
print(f"Risk Category: {risk_category}")

# 8. Learning Curves - understand if more data would help
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a learning curve plot for a model."""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    return plt

# Process X_train_processed for learning curve
X_train_processed = preprocessor.fit_transform(X_train)

print("\nGenerating learning curves...")

# Learning curve for the best model type
if best_model_name == 'Random Forest':
    lc_estimator = RandomForestClassifier(**rf_best.named_steps['classifier'].get_params())
elif best_model_name == 'Gradient Boosting':
    lc_estimator = GradientBoostingClassifier(**gb_best.named_steps['classifier'].get_params())
else:  # XGBoost
    lc_estimator = xgb.XGBClassifier(**xgb_best.named_steps['classifier'].get_params())

# Plot learning curve
lc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plot_learning_curve(lc_estimator, X_train_processed, y_train, 
                   title=f"Learning Curve ({best_model_name})", cv=lc_cv, n_jobs=-1)
plt.tight_layout()
plt.savefig('learning_curve.png')

print("Learning curve generated and saved.")
print("\nAnalysis complete. All visualizations have been saved.")
```

## Key Insights from the Implementation

### Performance Comparison

In our credit risk assessment example, we implemented three ensemble methods:
1. **Random Forest** (a bagging method)
2. **Gradient Boosting** (the classic implementation)
3. **XGBoost** (an optimized boosting implementation)

Key observations from the model comparison:

- **Accuracy**: All ensemble methods typically achieve high accuracy on this task, with boosting methods often slightly outperforming Random Forest.
- **Precision vs. Recall Trade-off**: Boosting methods often achieve better precision (fewer false positives), while Random Forest may have better recall (identifying more actual defaults).
- **AUC-ROC**: This is a robust metric for imbalanced classification problems like credit risk. XGBoost and Gradient Boosting typically achieve higher AUC than Random Forest.
- **Training Time**: Random Forest trains faster than boosting methods due to its parallel nature.
- **Hyperparameter Sensitivity**: Boosting methods (especially Gradient Boosting) are more sensitive to hyperparameter settings than Random Forest.

### Feature Importance Analysis

The feature importance analysis reveals:

- **Consistency Across Models**: Critical features like credit score, debt-to-income ratio, and number of late payments are identified as important by all models.
- **Differences in Ranking**: Each algorithm calculates feature importance differently, resulting in some variation in the ranking.
- **Random Forest**: Tends to give more balanced importance scores across features.
- **Boosting Methods**: Often concentrate more importance on a smaller subset of features.

### SHAP Analysis

The SHAP (SHapley Additive exPlanations) values provide deeper insights:
- They show how each feature contributes to individual predictions
- They reveal feature interactions that may not be apparent from simple importance measures
- They allow us to understand whether a feature increases or decreases the default probability

## Advanced Ensemble Techniques

### Stacking Ensemble

Stacking combines multiple models by training a meta-learner to predict based on the outputs of base models.

```python
# Example of stacking ensemble
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42))
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# This would be part of a pipeline with preprocessor
# stacking_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', stacking_clf)
# ])
```

### Voting Ensemble

Voting combines predictions from multiple models through majority voting (for classification) or averaging (for regression).

```python
# Example of voting ensemble
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft'  # 'hard' for majority voting, 'soft' for weighted probabilities
)

# This would be part of a pipeline with preprocessor
# voting_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', voting_clf)
# ])
```

## Best Practices for Ensemble Methods

### Random Forest Best Practices

1. **Number of Trees (n_estimators)**:
   - Start with 100-500 trees
   - More trees generally improve performance but with diminishing returns
   - Monitor out-of-bag error to determine optimal number

2. **Tree Depth (max_depth)**:
   - Control tree complexity to prevent overfitting
   - Consider setting a maximum depth or tuning min_samples_split/min_samples_leaf
   - Deeper trees can capture more complex patterns but are more prone to overfitting

3. **Feature Selection (max_features)**:
   - Default 'sqrt' (classification) or 'auto' (regression) often works well
   - Increasing max_features increases model strength but decreases diversity
   - Consider tuning this parameter for optimal performance

4. **Bootstrap Sampling (bootstrap and max_samples)**:
   - Enable bootstrap (default) to create diverse trees
   - Consider setting max_samples to control the size of bootstrap samples
   - Using bootstrap=False with limited samples may reduce diversity

### Gradient Boosting Best Practices

1. **Learning Rate and Trees (learning_rate and n_estimators)**:
   - Use a small learning rate (0.01-0.1) with more trees
   - These parameters work together: lower learning rate requires more trees
   - Consider early stopping to determine optimal number of trees

2. **Tree Depth (max_depth)**:
   - Keep trees shallow (3-8 levels) to prevent overfitting
   - Shallow trees make better weak learners for boosting
   - Even max_depth=1 (decision stumps) can work well with enough trees

3. **Subsampling (subsample)**:
   - Values less than 1.0 introduce randomness and help prevent overfitting
   - Common values are 0.8-1.0
   - Stochastic Gradient Boosting (subsample < 1.0) often performs better than standard

4. **Learning Rate Shrinkage**:
   - Start with a small learning rate (0.01-0.1)
   - Smaller values require more trees but often result in better performance
   - Too small can lead to underfitting and slow training

### XGBoost-Specific Best Practices

1. **Regularization Parameters (reg_alpha and reg_lambda)**:
   - L1 (reg_alpha) and L2 (reg_lambda) regularization control model complexity
   - Higher values can help prevent overfitting
   - Start with default values and tune if necessary

2. **Column and Row Sampling (colsample_bytree and subsample)**:
   - Introducing randomness helps prevent overfitting
   - Common values are 0.5-1.0
   - These can be particularly helpful with large, high-dimensional datasets

3. **Tree Construction Algorithm (tree_method)**:
   - 'exact' for small datasets, 'approx' or 'hist' for larger ones
   - 'gpu_hist' for GPU acceleration
   - Can significantly impact training speed

4. **Handling Missing Values (missing)**:
   - XGBoost can handle missing values natively
   - Set missing parameter to the value that represents missing in your data

### General Ensemble Best Practices

1. **Quality Over Quantity**:
   - Focus on creating diverse, high-quality base models
   - A few well-tuned, diverse models often outperform many similar ones

2. **Preserve Independent Errors**:
   - Ensure base models make different types of errors
   - Use different algorithms, features, or subsets of data

3. **Balance Bias and Variance**:
   - Combine low-bias/high-variance models with high-bias/low-variance ones
   - This helps achieve better overall bias-variance tradeoff

4. **Cross-Validation**:
   - Use robust cross-validation to tune hyperparameters
   - Consider nested cross-validation for unbiased performance estimation

5. **Monitor Overfitting**:
   - Track performance on validation data during training
   - Use early stopping when possible
   - Apply appropriate regularization techniques

## When to Use Which Ensemble Method

### Use Random Forests When:

- You need a robust, easy-to-tune algorithm
- Training speed is important (can be parallelized)
- You want balanced feature importance
- The dataset has many irrelevant features
- You need a good out-of-the-box performance with minimal tuning
- You're concerned about overfitting

### Use Gradient Boosting When:

- You need maximum predictive performance
- You have the time and resources to tune hyperparameters
- The dataset isn't too large (training is sequential)
- You want to squeeze out every bit of signal from the data
- You're willing to spend time on hyperparameter optimization

### Use XGBoost/LightGBM/CatBoost When:

- You need the benefits of Gradient Boosting with better performance
- You're working with large datasets
- You need faster training than traditional Gradient Boosting
- You want advanced regularization options
- You need robust handling of missing values (XGBoost)
- You have many categorical features (CatBoost)

## Conclusion

Ensemble methods like Random Forests and Gradient Boosting represent some of the most powerful tools in the machine learning arsenal. They combine the predictions of multiple models to achieve better performance than any single model could achieve alone.

The key insights to remember are:

1. **Random Forests** use bagging (bootstrap sampling) and random feature selection to create a diverse set of trees trained in parallel. They are robust, less prone to overfitting, and require minimal tuning.

2. **Gradient Boosting** methods train trees sequentially, with each tree correcting the errors of previous ones. They often achieve state-of-the-art performance but require more careful tuning to prevent overfitting.

3. **Advanced Implementations** like XGBoost, LightGBM, and CatBoost offer optimized versions of gradient boosting with additional features like efficient memory usage, sparse-aware split finding, and built-in handling of categorical variables.

4. **Model Interpretability** tools like feature importance and SHAP values help explain predictions from these otherwise "black box" models, making them more useful in practice.

By understanding how these ensemble methods work and when to apply each, data scientists can effectively tackle a wide range of machine learning problems from credit risk assessment to medical diagnosis, customer churn prediction, and beyond.

## Next Learning Steps

To deepen your understanding of ensemble methods:

1. **Explore Advanced Ensemble Techniques**:
   - Boosted Trees with constraints (e.g., Monotonic Constraints in XGBoost)
   - Deep Forest / Multi-Grained Cascade Forest
   - Feature-weighted Linear Stacking

2. **Study Theoretical Foundations**:
   - Bias-Variance Tradeoff in Ensemble Learning
   - Strength-Correlation Framework for Random Forests
   - Mathematical Foundations of Boosting Algorithms

3. **Implement Custom Ensembles**:
   - Create your own ensemble structures
   - Implement weighted averaging based on model confidence
   - Develop specialized ensembles for specific domains

4. **Explore Model Interpretation Deeply**:
   - Partial Dependence Plots
   - Individual Conditional Expectation Plots
   - Advanced SHAP Applications

5. **Apply to Specialized Domains**:
   - Time Series Forecasting with Ensemble Methods
   - Imbalanced Classification Problems
   - Multi-label and Multi-class Problems

By mastering ensemble methods, you'll add some of the most powerful and widely applicable tools to your machine learning toolkit, enabling you to solve complex real-world problems with greater accuracy and reliability.
