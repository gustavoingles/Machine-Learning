import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
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
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

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
results_df.to_csv("results/python_results.csv", index=False)
print("\nResults saved to results/python_results.csv")
