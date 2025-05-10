# Common Machine Learning Algorithms: Linear Regression, Logistic Regression, and Decision Trees

## Introduction

Machine learning algorithms can be categorized based on their learning approach, output type, and functionality. This document focuses on three fundamental algorithms that form the building blocks of many machine learning applications: linear regression, logistic regression, and decision trees. We'll explore each algorithm's theory, implementation, and practical applications with real-world examples.

## Linear Regression

### Theory and Concept

Linear regression is one of the oldest and most widely used supervised learning algorithms for predicting continuous values. The core idea is to establish a linear relationship between input features and a continuous output variable by fitting a linear equation to the observed data.

**Mathematical Formulation:**

For a dataset with n features, the linear regression model can be expressed as:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- y is the predicted value (dependent variable)
- β₀ is the y-intercept (bias term)
- β₁, β₂, ..., βₙ are the coefficients (weights) for each feature
- x₁, x₂, ..., xₙ are the feature values (independent variables)
- ε is the error term

The goal is to find the optimal values for the coefficients (β) that minimize the sum of squared differences between predicted and actual values. This is typically done using Ordinary Least Squares (OLS), which minimizes the cost function:

```
J(β) = Σ(y_actual - y_predicted)² = Σ(y_actual - (β₀ + β₁x₁ + ... + βₙxₙ))²
```

### Types of Linear Regression

1. **Simple Linear Regression**: Uses a single feature to predict the target variable
2. **Multiple Linear Regression**: Uses multiple features to predict the target variable
3. **Polynomial Regression**: Extends linear regression by adding polynomial terms (x², x³, etc.)
4. **Ridge Regression**: Adds L2 regularization to prevent overfitting
5. **Lasso Regression**: Adds L1 regularization to encourage sparse solutions

### Real-World Example: House Price Prediction

Let's implement a linear regression model to predict house prices based on features like square footage, number of bedrooms, etc.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Set random seed for reproducibility
np.random.seed(42)

# Load the California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Display basic information about the dataset
print("Dataset shape:", X.shape)
print("Feature names:", housing.feature_names)
print("Target (house prices) statistics:")
print(f"Min: ${y.min()*100000:.2f}, Max: ${y.max()*100000:.2f}, Mean: ${y.mean()*100000:.2f}")

# Let's look at the first few rows of data
print("\nFirst 5 rows of data:")
X_with_target = X.copy()
X_with_target['Price'] = y
print(X_with_target.head())

# Explore the relationship between features and target
plt.figure(figsize=(12, 10))
correlation_matrix = X_with_target.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Select features with the highest correlation to price
plt.figure(figsize=(10, 6))
plt.scatter(X['MedInc'], y, alpha=0.5)
plt.title('House Price vs. Median Income')
plt.xlabel('Median Income (scaled)')
plt.ylabel('House Price (in $100,000s)')
plt.tight_layout()
plt.savefig('price_vs_income.png')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nModel Performance:")
print(f"Training MSE: {train_mse:.4f}, RMSE: ${np.sqrt(train_mse)*100000:.2f}")
print(f"Testing MSE: {test_mse:.4f}, RMSE: ${np.sqrt(test_mse)*100000:.2f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

# Visualize the model coefficients
feature_importance = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('linear_regression_coefficients.png')

# Visualize predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Prices (in $100,000s)')
plt.ylabel('Predicted Prices (in $100,000s)')
plt.title('Actual vs. Predicted House Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')

# Predict for a new house
def predict_house_price(features, scaler, model):
    """
    Predict the price of a house based on its features.
    
    Args:
        features: Array of features in the same order as the training data
        scaler: Fitted StandardScaler
        model: Trained linear regression model
        
    Returns:
        Predicted price in dollars
    """
    features_scaled = scaler.transform([features])
    predicted_value = model.predict(features_scaled)[0]
    return predicted_value * 100000  # Convert to dollars

# Example: Predict price for a new house
new_house = [8.3252, 41.0, 6.984127, 1.023, 322.0, 2.555556, 37.88, -122.23]  # Sample values
predicted_price = predict_house_price(new_house, scaler, model)
print(f"\nPredicted price for the new house: ${predicted_price:.2f}")

# Residual analysis
residuals = y_test - y_test_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residual Value')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('residuals_distribution.png')

plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.tight_layout()
plt.savefig('residuals_vs_predicted.png')
```

### Interpretation of Results

- **Model Coefficients**: Each coefficient represents the expected change in the house price for a one-unit increase in the corresponding feature, holding all other features constant. For example, if the coefficient for 'MedInc' (median income) is 0.5, it means that for every 1-unit increase in median income, we expect the house price to increase by $50,000 (since our target is in units of $100,000s).

- **R² Score**: The R² value indicates the proportion of variance in the dependent variable that can be predicted from the independent variables. An R² of 0.70 means that 70% of the variance in house prices can be explained by our features.

- **Residual Analysis**: Examining the residuals helps us validate the assumptions of linear regression:
  - Residuals should be normally distributed around zero
  - Residuals should show no pattern when plotted against predicted values
  - If the residuals show patterns, it might indicate non-linear relationships or missing features

### Advantages of Linear Regression

1. **Simplicity and Interpretability**: Easy to understand and explain the relationship between features and target
2. **Efficiency**: Trains quickly and requires minimal computational resources
3. **Well-studied Properties**: Strong statistical foundation with well-understood behaviors
4. **Feature Importance**: Coefficients provide a clear measure of feature importance

### Limitations of Linear Regression

1. **Linearity Assumption**: Assumes a linear relationship between features and target
2. **Sensitivity to Outliers**: Outliers can significantly affect the model
3. **Feature Independence**: Assumes features are not highly correlated (multicollinearity issues)
4. **Constant Variance**: Assumes the error variance is constant (homoscedasticity)

## Logistic Regression

### Theory and Concept

Despite its name, logistic regression is a classification algorithm, not a regression algorithm. It's used to predict the probability that an instance belongs to a particular class. The algorithm applies the logistic function to a linear combination of features to transform the output to a probability value between 0 and 1.

**Mathematical Formulation:**

The logistic regression model can be expressed as:

```
p(y=1|x) = σ(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)
```

Where:
- p(y=1|x) is the probability that the instance belongs to class 1 given features x
- σ is the sigmoid function: σ(z) = 1 / (1 + e^(-z))
- β₀, β₁, ..., βₙ are the model parameters
- x₁, x₂, ..., xₙ are the feature values

The decision boundary is the set of points where p(y=1|x) = 0.5, which corresponds to:
```
β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ = 0
```

The model is trained by maximizing the likelihood of the observed data, typically using methods like gradient descent to minimize the log loss (cross-entropy).

### Types of Logistic Regression

1. **Binary Logistic Regression**: Predicts one of two possible outcomes (e.g., spam/not spam)
2. **Multinomial Logistic Regression**: Predicts one of multiple classes with no natural ordering
3. **Ordinal Logistic Regression**: Predicts one of multiple ordered classes
4. **Regularized Logistic Regression**: Adds L1 or L2 regularization to prevent overfitting

### Real-World Example: Customer Churn Prediction

Let's implement a logistic regression model to predict customer churn (whether a customer will leave a service or not).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Load a sample customer churn dataset
# For this example, we'll create synthetic data
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.8, 0.2],  # Imbalanced classes - 80% non-churn, 20% churn
    random_state=42
)

# Create a DataFrame with meaningful feature names
feature_names = [
    'MonthlyCharges', 'TotalCharges', 'Tenure', 'ContractLength',
    'OnlineService', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'PaymentMethod'
]
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.Series(y, name='Churn')

# Add the target to the DataFrame for exploration
data = X_df.copy()
data['Churn'] = y_df

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("Feature names:", feature_names)
print("\nClass distribution:")
print(data['Churn'].value_counts())
print(f"Churn rate: {data['Churn'].mean()*100:.2f}%")

# Explore the data
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('churn_correlation_matrix.png')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print(f"Training set churn rate: {y_train.mean()*100:.2f}%")
print(f"Testing set churn rate: {y_test.mean()*100:.2f}%")

# Create a pipeline with preprocessing and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Extract the trained logistic regression model
logistic_model = pipeline.named_steps['classifier']

# Make predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print("\nModel Performance:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"AUC-ROC: {test_auc:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('churn_confusion_matrix.png')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('churn_roc_curve.png')

# Feature importance (coefficients)
scaler = pipeline.named_steps['scaler']
scaled_coef = logistic_model.coef_[0] * scaler.scale_

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': scaled_coef,
    'Abs_Coefficient': np.abs(scaled_coef)
}).sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient Value (Impact on Log-Odds of Churn)')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('churn_coefficients.png')

# Probability calibration
plt.figure(figsize=(10, 6))
plt.hist(y_test_prob, bins=20, range=(0, 1), alpha=0.8, color='skyblue')
plt.xlabel('Predicted Probability of Churn')
plt.ylabel('Count')
plt.title('Distribution of Predicted Probabilities')
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.legend()
plt.tight_layout()
plt.savefig('churn_probability_distribution.png')

# Prediction for a new customer
def predict_churn(features, pipeline):
    """
    Predict whether a customer will churn based on their features.
    
    Args:
        features: Array of features in the same order as the training data
        pipeline: Trained logistic regression pipeline
        
    Returns:
        Predicted churn status and probability
    """
    features_df = pd.DataFrame([features], columns=feature_names)
    churn_prob = pipeline.predict_proba(features_df)[0, 1]
    churn_pred = pipeline.predict(features_df)[0]
    
    return churn_pred, churn_prob

# Example: Predict churn for a new customer
new_customer = [70, 2000, 12, -0.5, 1.2, -0.8, 0.5, 0.7, 1.0, -0.3]  # Sample values
churn_prediction, churn_probability = predict_churn(new_customer, pipeline)

print("\nNew Customer Churn Prediction:")
print(f"Churn Status: {'Will Churn' if churn_prediction == 1 else 'Will Not Churn'}")
print(f"Churn Probability: {churn_probability:.2f}")

# Optional: Perform Recursive Feature Elimination to find the most important features
rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=5)
rfe.fit(X_train, y_train)

print("\nTop 5 features selected by RFE:")
selected_features = [feature for feature, selected in zip(feature_names, rfe.support_) if selected]
print(selected_features)
```

### Interpretation of Results

- **Model Coefficients**: In logistic regression, coefficients represent the change in the log-odds of the outcome for a one-unit increase in the predictor. Positive coefficients increase the probability of churn, while negative coefficients decrease it.

- **Performance Metrics**:
  - **Accuracy**: The proportion of correct predictions
  - **Precision**: The proportion of predicted positives that are actually positive (true positives / (true positives + false positives))
  - **Recall**: The proportion of actual positives correctly identified (true positives / (true positives + false negatives))
  - **F1 Score**: The harmonic mean of precision and recall
  - **AUC-ROC**: Area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes

- **Threshold Selection**: Logistic regression produces probabilities, which are converted to class predictions using a threshold (default is 0.5). Adjusting this threshold allows balancing between precision and recall based on business needs.

### Advantages of Logistic Regression

1. **Interpretability**: Coefficients have clear interpretations as log-odds
2. **Probabilistic Output**: Provides probabilities rather than just class labels
3. **Efficiency**: Trains quickly and scales well to large datasets
4. **Regularization Options**: Can incorporate L1/L2 regularization to prevent overfitting
5. **No Distributional Assumptions**: Does not assume normal distribution of features

### Limitations of Logistic Regression

1. **Linear Decision Boundary**: Cannot capture complex non-linear relationships without feature engineering
2. **Feature Independence**: Assumes features are not highly correlated
3. **Limited to Probability Estimation**: Only models the probability of the outcome, not directly the outcome itself
4. **Data Separation Issues**: Can fail to converge with perfectly separable classes

## Decision Trees

### Theory and Concept

Decision trees are versatile machine learning algorithms that can perform both classification and regression tasks. They create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. The tree structure represents a flowchart-like diagram where each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label or value.

**How Decision Trees Work:**

1. **Feature Selection**: At each node, select the feature that best splits the data
2. **Split Point Selection**: Determine the optimal value to split on
3. **Recursive Partitioning**: Repeat the process for each child node
4. **Stopping Criteria**: Stop when a node contains only samples of one class, or when further splitting doesn't improve results

**Splitting Criteria:**

- **Classification Trees**:
  - Gini Impurity: Measures the probability of incorrect classification
  - Entropy/Information Gain: Measures the reduction in uncertainty
  
- **Regression Trees**:
  - Mean Squared Error (MSE): Measures the average squared difference between the observed and predicted values
  - Mean Absolute Error (MAE): Measures the average absolute difference between the observed and predicted values

### Types of Decision Trees

1. **Classification and Regression Trees (CART)**: Can be used for both tasks
2. **C4.5/C5.0**: Extensions of the ID3 algorithm with improvements
3. **Chi-square Automatic Interaction Detection (CHAID)**: Uses chi-square tests for splitting
4. **Conditional Inference Trees**: Uses statistical tests for splitting decisions
5. **Ensemble Methods**: Random Forests, Gradient Boosting use multiple trees together

### Real-World Example: Credit Approval

Let's implement a decision tree model to predict whether a credit application should be approved based on various applicant features.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# For this example, we'll create synthetic data for credit approval
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.7, 0.3],  # 70% approved, 30% denied
    random_state=42
)

# Create a DataFrame with meaningful feature names
# Numerical features
feature_names = [
    'Income', 'Debt', 'CreditScore', 'Age', 'EmploymentLength',
    'LoanAmount', 'PropertyValue', 'MonthlyExpenses', 'NumCreditLines', 'NumLatePays'
]
X_df = pd.DataFrame(X, columns=feature_names)

# Add categorical features (for demonstration)
n_samples = X_df.shape[0]
X_df['EmploymentStatus'] = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], size=n_samples, p=[0.7, 0.2, 0.1])
X_df['Education'] = np.random.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
X_df['MaritalStatus'] = np.random.choice(['Single', 'Married', 'Divorced'], size=n_samples, p=[0.4, 0.5, 0.1])

# Target variable: 1 = Approved, 0 = Denied
y_df = pd.Series(y, name='Approved')

# Combine features and target for exploration
data = X_df.copy()
data['Approved'] = y_df

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("Feature names:", X_df.columns.tolist())
print("\nClass distribution:")
print(data['Approved'].value_counts())
print(f"Approval rate: {data['Approved'].mean()*100:.2f}%")

# Look at basic statistics for numerical features
print("\nNumerical feature statistics:")
print(data[feature_names].describe())

# Distribution of categorical features
print("\nCategorical feature distributions:")
for cat_feature in ['EmploymentStatus', 'Education', 'MaritalStatus']:
    print(f"\n{cat_feature}:")
    print(data[cat_feature].value_counts(normalize=True) * 100)

# Split features into numerical and categorical
numerical_features = feature_names
categorical_features = ['EmploymentStatus', 'Education', 'MaritalStatus']

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Create a pipeline with preprocessing and decision tree
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train the model with default parameters
pipeline.fit(X_train, y_train)

# Make predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print("\nDefault Decision Tree Performance:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"AUC-ROC: {test_auc:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Denied', 'Approved'],
            yticklabels=['Denied', 'Approved'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('credit_confusion_matrix.png')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('credit_roc_curve.png')

# Hyperparameter tuning
param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters found by grid search:")
print(grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the tuned model
y_train_pred_tuned = best_model.predict(X_train)
y_test_pred_tuned = best_model.predict(X_test)
y_test_prob_tuned = best_model.predict_proba(X_test)[:, 1]

# Evaluate the tuned model
train_accuracy_tuned = accuracy_score(y_train, y_train_pred_tuned)
test_accuracy_tuned = accuracy_score(y_test, y_test_pred_tuned)
test_precision_tuned = precision_score(y_test, y_test_pred_tuned)
test_recall_tuned = recall_score(y_test, y_test_pred_tuned)
test_f1_tuned = f1_score(y_test, y_test_pred_tuned)
test_auc_tuned = roc_auc_score(y_test, y_test_prob_tuned)

print("\nTuned Decision Tree Performance:")
print(f"Training Accuracy: {train_accuracy_tuned:.4f}")
print(f"Testing Accuracy: {test_accuracy_tuned:.4f}")
print(f"Precision: {test_precision_tuned:.4f}")
print(f"Recall: {test_recall_tuned:.4f}")
print(f"F1 Score: {test_f1_tuned:.4f}")
print(f"AUC-ROC: {test_auc_tuned:.4f}")

# Extract the tuned decision tree
tree_model = best_model.named_steps['classifier']

# Get feature names after preprocessing (for visualization)
cat_encoder = best_model.named_steps['preprocessor'].transformers_[1][1]
encoded_cats = cat_encoder.get_feature_names_out(categorical_features)
feature_names_after_transformation = numerical_features + list(encoded_cats)

# Visualize the decision tree (limited to depth 3 for clarity)
plt.figure(figsize=(20, 12))
plot_tree(tree_model, 
          max_depth=3, 
          feature_names=feature_names_after_transformation,
          class_names=['Denied', 'Approved'],
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title('Decision Tree for Credit Approval (Limited to Depth 3)')
plt.tight_layout()
plt.savefig('credit_decision_tree.png')

# Feature importance
importances = tree_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("\nFeature ranking:")
for f in range(min(10, len(feature_names_after_transformation))):
    print(f"{f+1}. {feature_names_after_transformation[indices[f]]} ({importances[indices[f]]:.4f})")

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.bar(range(min(10, len(indices))), importances[indices[:10]], align='center')
plt.xticks(range(min(10, len(indices))), [feature_names_after_transformation[i] for i in indices[:10]], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances in the Decision Tree')
plt.tight_layout()
plt.savefig('credit_feature_importance.png')

# Print a text representation of the tree
tree_rules = export_text(tree_model, feature_names=feature_names_after_transformation, max_depth=3)
print("\nDecision Tree Rules (Limited to Depth 3):")
print(tree_rules)

# Prediction for a new applicant
def predict_credit_approval(features, model):
    """
    Predict whether a credit application will be approved.
    
    Args:
        features: Dictionary of features
        model: Trained pipeline model
        
    Returns:
        Prediction and probability
    """
    # Convert dictionary to DataFrame
    features_df = pd.DataFrame([features])
    
    # Make prediction
    approval_prob = model.predict_proba(features_df)[0, 1]
    approval_pred = model.predict(features_df)[0]
    
    return approval_pred, approval_prob

# Example new applicant
new_applicant = {
    'Income': 75000,
    'Debt': 15000,
    'CreditScore': 720,
    'Age': 35,
    'EmploymentLength': 5,
    'LoanAmount': 200000,
    'PropertyValue': 250000,
    'MonthlyExpenses': 3000,
    'NumCreditLines': 3,
    'NumLatePays': 0,
    'EmploymentStatus': 'Employed',
    'Education': 'Bachelor',
    'MaritalStatus': 'Married'
}

# Transform the applicant data to match our feature format
new_applicant_transformed = {key: [np.random.normal()] if key in numerical_features else value 
                            for key, value in new_applicant.items()}

# Make prediction
approval_prediction, approval_probability = predict_credit_approval(new_applicant_transformed, best_model)

print("\nNew Applicant Prediction:")
print(f"Credit Status: {'Approved' if approval_prediction == 1 else 'Denied'}")
print(f"Approval Probability: {approval_probability:.2f}")
```

### Interpretation of Results

- **Tree Structure**: Each node in the decision tree represents a decision based on a feature value, creating paths that lead to final predictions.

- **Feature Importance**: The importance of each feature is determined by how much it contributes to reducing impurity (Gini or entropy) across all splits in the tree.

- **Decision Paths**: Following a path from the root to a leaf gives a clear sequence of decisions that lead to a particular prediction, making the model highly interpretable.

- **Performance Metrics**:
  - Similar to logistic regression, we evaluate using accuracy, precision, recall, F1-score, and AUC-ROC
  - The tree's depth and complexity significantly impact its performance and tendency to overfit

- **Hyperparameter Tuning**: Parameters like maximum depth, minimum samples per split, and minimum samples per leaf control the tree's complexity and ability to generalize.

### Advantages of Decision Trees

1. **Interpretability**: Decision trees are easy to visualize and explain, with clear decision rules
2. **Non-parametric**: No assumptions about data distribution
3. **Handles Mixed Data Types**: Can work with both numerical and categorical features
4. **Minimal Data Preparation**: No need for feature scaling or normalization
5. **Handles Missing Values**: Can work with missing data effectively
6. **Captures Non-linear Relationships**: Can model complex patterns without explicit feature engineering

### Limitations of Decision Trees

1. **Overfitting**: Prone to creating overly complex trees that don't generalize well
2. **Instability**: Small changes in data can lead to completely different trees
3. **Biased Toward Features with More Levels**: Can be biased toward features with more unique values
4. **Difficulty with Unbalanced Data**: May create biased trees if some classes dominate
5. **Limited Expressiveness**: Single trees may not capture certain relationships that more complex models can

## Comparison of Algorithms

| Aspect | Linear Regression | Logistic Regression | Decision Trees |
|--------|-------------------|---------------------|---------------|
| **Type** | Regression | Classification | Both |
| **Output** | Continuous values | Probabilities (0-1) | Classes or values |
| **Function** | Linear function | Sigmoid function | Piecewise function |
| **Interpretability** | High | High | Very high |
| **Data Preparation** | Scaling helpful | Scaling helpful | Minimal required |
| **Handles Categorical Data** | Requires encoding | Requires encoding | Naturally handles categories |
| **Feature Importance** | Coefficient values | Coefficient values | Impurity reduction |
| **Handles Missing Values** | Poorly | Poorly | Well |
| **Captures Non-linearity** | Poorly (without transformations) | Poorly (without transformations) | Well |
| **Prone to Overfitting** | Low | Moderate | High |
| **Training Speed** | Fast | Fast | Moderate |
| **Prediction Speed** | Very fast | Very fast | Fast |
| **Best Used For** | Continuous target prediction, relationship understanding | Binary/multi-class classification, probability estimation | Both classification and regression, rule extraction |

## Best Practices and Recommendations

### When to Use Linear Regression
- When predicting continuous values
- When the relationship between features and target is approximately linear
- When interpretability is important
- When computation speed is critical
- Example applications: Price prediction, sales forecasting, trend analysis

### When to Use Logistic Regression
- For binary classification problems
- When probability estimates are needed
- When the decision boundary is approximately linear
- When interpreting feature effects is important
- Example applications: Spam detection, disease diagnosis, customer churn prediction

### When to Use Decision Trees
- When interpretability through decision rules is crucial
- When features interact in complex, non-linear ways
- When the data contains mixed types (numerical and categorical)
- When handling missing values without preprocessing
- Example applications: Credit scoring, medical diagnosis, customer segmentation

### General Recommendations
1. **Start Simple**: Begin with simpler models before trying complex ones
2. **Feature Engineering**: Create meaningful features based on domain knowledge
3. **Cross-Validation**: Use cross-validation to ensure reliable performance estimates
4. **Regularization**: Apply appropriate regularization techniques to prevent overfitting
5. **Ensemble Methods**: Consider ensemble methods like Random Forests or Gradient Boosting for improved performance
6. **Interpret Results**: Always interpret the model in the context of the problem domain
7. **Monitor Deployment**: Track model performance in production and retrain as needed

## Conclusion

Linear regression, logistic regression, and decision trees form the foundation of many machine learning applications. Each algorithm has its strengths, limitations, and appropriate use cases:

- **Linear Regression** offers a straightforward approach to predicting continuous values with high interpretability.
- **Logistic Regression** provides an elegant solution for classification problems with probabilistic outputs.
- **Decision Trees** deliver highly interpretable models that can capture complex relationships in both classification and regression tasks.

Understanding these fundamental algorithms provides a solid basis for tackling more advanced techniques and real-world machine learning challenges. The choice of algorithm should be guided by the nature of the problem, the characteristics of the data, and the specific requirements of the application.

## Next Learning Steps

1. **Ensemble Methods**: Explore Random Forests, Gradient Boosting, and other ensemble techniques that build upon these basic algorithms
2. **Advanced Regularization**: Learn more about regularization methods like elastic net and early stopping
3. **Model Evaluation**: Deepen your understanding of evaluation metrics and validation techniques
4. **Feature Selection**: Study techniques for identifying the most relevant features
5. **Hyperparameter Tuning**: Master approaches to optimize model parameters systematically
6. **Practical Applications**: Apply these algorithms to real-world datasets in different domains

Remember that mastering these fundamental algorithms provides an excellent foundation for understanding more complex machine learning techniques.
