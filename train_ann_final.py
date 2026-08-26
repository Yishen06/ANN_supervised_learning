"""
Heart Disease Prediction using Artificial Neural Network (ANN) — FINAL VERSION
AI Assignment - Machine Learning (Supervised) - ANN Method

Dataset: synthetic_heart_disease_dataset.csv (50,000 patients, 21 columns)

Pipeline:
1. Load & inspect data
2. Encode categorical text columns
3. Train/test split (BEFORE imputing/scaling, to avoid data leakage)
4. Impute missing values (Alcohol_Intake) using training-set statistics only
5. Scale features
6. Hyperparameter tuning (compare several ANN architectures via cross-validation)
7. Train final model with the best architecture
8. Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
9. Predict on a new, unseen patient (demo-ready function)
10. Save model + preprocessing objects for reuse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 1. Load data
# -----------------------------
try:
    df = pd.read_csv("synthetic_heart_disease_dataset.csv")
except FileNotFoundError:
    raise SystemExit(
        "ERROR: 'synthetic_heart_disease_dataset.csv' not found. "
        "Make sure the CSV is in the same folder as this script."
    )

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nTarget distribution:\n", df["Heart_Disease"].value_counts(normalize=True))

# -----------------------------
# 2. Encode categorical text columns
# -----------------------------
# Binary text columns
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Nominal / ordinal text columns (label-encoded; simple & effective for tree/ANN inputs)
smoking_map = {"Never": 0, "Former": 1, "Current": 2}
df["Smoking"] = df["Smoking"].map(smoking_map)

alcohol_map = {"Low": 0, "Moderate": 1, "High": 2}   # NaN stays NaN, handled by imputer later
df["Alcohol_Intake"] = df["Alcohol_Intake"].map(alcohol_map)

activity_map = {"Sedentary": 0, "Moderate": 1, "Active": 2}
df["Physical_Activity"] = df["Physical_Activity"].map(activity_map)

diet_map = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
df["Diet"] = df["Diet"].map(diet_map)

stress_map = {"Low": 0, "Medium": 1, "High": 2}
df["Stress_Level"] = df["Stress_Level"].map(stress_map)

print("\nAfter encoding, dtypes:\n", df.dtypes)

# -----------------------------
# 3. Correlation check (sanity check the data has real signal)
# -----------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("\nSaved correlation_heatmap.png")

# -----------------------------
# 4. Train/test split (BEFORE imputing/scaling — avoids data leakage)
# -----------------------------
X = df.drop(columns=["Heart_Disease"])
y = df["Heart_Disease"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------
# 5. Impute missing values (Alcohol_Intake) using training data only
# -----------------------------
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test), columns=X_test.columns, index=X_test.index
)
print("Remaining missing values (train):", X_train_imputed.isnull().sum().sum())

# -----------------------------
# 6. Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# -----------------------------
# 7. Hyperparameter tuning: compare ANN architectures via 5-fold CV
# -----------------------------
print("\n--- Hyperparameter tuning (this may take a minute) ---")
architectures = [(8,), (16,), (32,), (16, 8), (32, 16), (64, 32)]
cv_scores = []
for arch in architectures:
    mlp = MLPClassifier(
        hidden_layer_sizes=arch, activation="relu", solver="adam",
        max_iter=300, random_state=RANDOM_STATE, early_stopping=True,
    )
    scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5, scoring="f1")
    cv_scores.append(scores.mean())
    print(f"Architecture {arch}: CV F1 = {scores.mean():.4f}")

best_idx = int(np.argmax(cv_scores))
best_arch = architectures[best_idx]
print(f"\nBest architecture: {best_arch} (CV F1 = {max(cv_scores):.4f})")

plt.figure(figsize=(8, 4))
plt.bar([str(a) for a in architectures], cv_scores, color="#4C72B0")
plt.title("ANN Cross-Validation F1 Score vs Hidden Layer Architecture")
plt.xlabel("Hidden Layer Sizes")
plt.ylabel("Cross-Validated F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("hyperparameter_tuning.png", dpi=150)
plt.close()
print("Saved hyperparameter_tuning.png")

# -----------------------------
# 8. Train final model with best architecture
# -----------------------------
final_model = MLPClassifier(
    hidden_layer_sizes=best_arch, activation="relu", solver="adam",
    max_iter=500, random_state=RANDOM_STATE, early_stopping=True,
)
final_model.fit(X_train_scaled, y_train)

y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)[:, 1]

# Training loss curve
plt.figure(figsize=(7, 4))
plt.plot(final_model.loss_curve_, color="#4C72B0")
plt.title(f"ANN Training Loss Curve (architecture = {best_arch})")
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("training_loss_curve.png", dpi=150)
plt.close()
print("Saved training_loss_curve.png")

# -----------------------------
# 9. Evaluation
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n===== Test Set Performance =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")


# Confusion matrix + ROC curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#4C72B0")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig("ann_results.png", dpi=150)
plt.close()
print("Saved ann_results.png")

# -----------------------------
# 10. Predict on a new, unseen patient (demo-ready function)
# -----------------------------


# -----------------------------
# 11. Save model + preprocessing objects for reuse (e.g. in a demo UI)
# -----------------------------
joblib.dump(final_model, "heart_disease_ann_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(feature_names, "feature_names.pkl")
print("\nModel and preprocessing objects saved.")
