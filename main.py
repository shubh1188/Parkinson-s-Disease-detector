import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score as ras, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')  

# Load dataset
df = pd.read_csv('parkinson_disease.csv')

# Dataset info
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nDataset description:")
print(df.describe())

# Preprocessing
df = df.groupby('id').mean().reset_index()  # Aggregate rows by ID
df.drop('id', axis=1, inplace=True)  # Drop ID column

# Correlation-based feature selection
columns = list(df.columns)
filtered_columns = []
for col in columns:
    if col == 'class':
        filtered_columns.append(col)
        continue

    correlated = False
    for col1 in filtered_columns:
        if col1 != 'class':
            if df[col].corr(df[col1]) > 0.7:
                correlated = True
                break
    if not correlated:
        filtered_columns.append(col)

df = df[filtered_columns]
print("Filtered dataset shape:", df.shape)

# Feature selection using SelectKBest
X = df.drop('class', axis=1)
y = df['class']

X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k=min(5, X.shape[1]))  # Select up to 5 best features
X_new = selector.fit_transform(X_norm, y)
selected_features = X.columns[selector.get_support()]
df = pd.DataFrame(X_new, columns=selected_features)
df['class'] = y
print("Dataset after feature selection shape:", df.shape)

# Plot class distribution
class_counts = df['class'].value_counts()
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=['skyblue', 'orange'])
plt.title("Class Distribution")
plt.show()

# Split dataset (stratified)
features = df.drop('class', axis=1)
target = df['class']

X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, random_state=10, stratify=target
)
print(f"Train shape: {X_train.shape} Validation shape: {X_val.shape}")

# Oversample minority class in training set
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X_train, Y_train = ros.fit_resample(X_train, Y_train)
print(f"Balanced dataset shape: {X_train.shape}, {Y_train.shape}")

# Model training and evaluation
models = [LogisticRegression(), XGBClassifier(use_label_encoder=False, eval_metric='logloss'), SVC(probability=True)]
for model in models:
    model.fit(X_train, Y_train)
    print(f"\nModel: {model.__class__.__name__}")
    
    # Training AUC-ROC
    train_preds = model.predict_proba(X_train)[:, 1]
    print("Training AUC-ROC:", ras(Y_train, train_preds))
    
    # Validation AUC-ROC
    if len(Y_val.unique()) > 1:  # Ensure at least two classes in validation set
        val_preds = model.predict_proba(X_val)[:, 1]
        print("Validation AUC-ROC:", ras(Y_val, val_preds))
    else:
        print("Validation AUC-ROC: Cannot compute AUC as only one class is present in the validation set.")

# Confusion matrix and classification report for Logistic Regression
if len(Y_val.unique()) > 1:
    ConfusionMatrixDisplay.from_estimator(models[0], X_val, Y_val, cmap='Blues')
    plt.title("Confusion Matrix for Logistic Regression")
    plt.show()

    print("\nClassification Report (Logistic Regression):")
    print(classification_report(Y_val, models[0].predict(X_val)))
else:
    print("\nConfusion matrix and classification report cannot be generated: Only one class in validation set.")

# Optional: Cross-validation for AUC-ROC
print("\nCross-Validation AUC-ROC Scores:")
for model in models:
    scores = cross_val_score(model, features, target, cv=5, scoring='roc_auc')
    print(f"{model.__class__.__name__}: {scores.mean():.3f} ± {scores.std():.3f}")
