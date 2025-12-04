import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import os

# 1. Load Data
FILE_PATH = "data/p33.xlsx"
print(f"Loading data from {FILE_PATH}...")
df = pd.read_excel(FILE_PATH)

# Drop 'id' if it exists, as it's not a feature
if 'id' in df.columns:
    df = df.drop('id', axis=1)

print("Data loaded. Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Preprocessing
# Identify columns
target_col = 'cancel'
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target_col]
numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != target_col]

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Separate X and y
X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)
print(f"Target encoded. Classes: {le.classes_}")

# Define Preprocessing Pipeline
# Numeric: Impute missing (mean) -> Scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical: Impute missing (most_frequent) -> OneHot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 3. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "kNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Neural Network": MLPClassifier(max_iter=3000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# 4. Evaluation Loop
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'auc': 'roc_auc',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0)
}

results = []

print("\nStarting Cross-Validation (10 folds)...")

for name, model in models.items():
    print(f"Evaluating {name}...")
    
    # Create full pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    
    # Store mean scores
    results.append({
        "Model": name,
        "Accuracy": np.mean(cv_results['test_accuracy']),
        "F1-Score": np.mean(cv_results['test_f1']),
        "AUC": np.mean(cv_results['test_auc']),
        "Precision": np.mean(cv_results['test_precision']),
        "Recall": np.mean(cv_results['test_recall'])
    })

# 5. Generate Report
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="AUC", ascending=False)

print("\n--- Comparative Results ---")
print(results_df.to_string(index=False))

# Save to CSV for easy inclusion in report
if not os.path.exists("results"):
    os.makedirs("results")
results_df.to_csv("results/python_results.csv", index=False)
print("\nResults saved to results/python_results.csv")

# 6. Visualization for Logistic Regression
print("\nGenerating visualizations for Logistic Regression...")
if not os.path.exists("plots"):
    os.makedirs("plots")

# Split data for visualization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression
lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# a) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/logistic_regression_confusion_matrix.png')
plt.close()
print("Saved plots/logistic_regression_confusion_matrix.png")

# b) ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plots/logistic_regression_roc_curve.png')
plt.close()
print("Saved plots/logistic_regression_roc_curve.png")

# c) Feature Coefficients
# Access the classifier step
classifier = lr_model.named_steps['classifier']
# Access the preprocessor step
preprocessor_step = lr_model.named_steps['preprocessor']

# Get feature names from OneHotEncoder
# Note: This depends on the order of transformers in ColumnTransformer
# Numerical features are first, then categorical
feature_names = numerical_cols.copy()
cat_encoder = preprocessor_step.named_transformers_['cat'].named_steps['onehot']
# Get feature names for categorical variables
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
feature_names.extend(cat_feature_names)

coefs = classifier.coef_[0]

# Create DataFrame for plotting
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.title('Feature Coefficients - Logistic Regression')
plt.tight_layout()
plt.savefig('plots/logistic_regression_coefficients.png')
plt.close()
print("Saved plots/logistic_regression_coefficients.png")
