import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ROC
# Q1

log_reg = LogisticRegression(
    C=1.0,
    max_iter = 1000,
    random_state=42

)
log_reg.fit(X_train, y_train)

log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
log_reg_auc = roc_auc_score(y_test, log_reg_probs)

print("Logistic Regression AUC:", log_reg_auc)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]

knn_auc = roc_auc_score(y_test, knn_probs)

print("KNN AUC:", knn_auc)

# KNN has a higher AUC (0.9394) than Logistic Regression (0.7060).
# This indicates that KNN is better at distinguishing between the
# positive and negative classes on this dataset and provides
# stronger overall classification performance.

# Q2

# Logistic Regression ROC
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, log_reg_probs)

# KNN ROC
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_probs)

plt.figure(figsize=(8, 6))

# Logistic Regression
plt.plot(
    fpr_log,
    tpr_log,
    label=f"Logistic Regression (AUC = {log_reg_auc:.3f})"
)

# KNN
plt.plot(
    fpr_knn,
    tpr_knn,
    label=f"KNN (AUC = {knn_auc:.3f})"
)

# Random classifier
plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")

plt.savefig("assignments_04/outputs/roc_comparison.png")
plt.show()

import numpy as np

# Logistic Regression
idx_log = np.argmin(np.abs(tpr_log - 0.80))
print("Logistic Regression")
print("TPR:", tpr_log[idx_log])
print("FPR:", fpr_log[idx_log])

# KNN
idx_knn = np.argmin(np.abs(tpr_knn - 0.80))
print("\nKNN")
print("TPR:", tpr_knn[idx_knn])
print("FPR:", fpr_knn[idx_knn])

# At approximately TPR = 0.80, KNN has a much lower FPR (0.04)
# compared with Logistic Regression (0.55). This means that if we
# need to detect about 80% of positive cases, KNN produces far fewer
# false alarms. KNN would be the better model when
# reducing false positives is important.

# Q3
from sklearn.metrics import f1_score

fpr_lr, tpr_lr, thresholds_lr = roc_curve(
    y_test,
    log_reg_probs
)

f1_scores = []

for threshold in thresholds_lr:
    y_pred = (log_reg_probs >= threshold).astype(int)
    
    f1 = f1_score(y_test, y_pred)
    
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds_lr[best_idx]
best_f1 = f1_scores[best_idx]
best_tpr = tpr_lr[best_idx]
best_fpr = fpr_lr[best_idx]

print("Optimal threshold:", best_threshold)
print("TPR:", best_tpr)
print("FPR:", best_fpr)
print("F1 score:", best_f1)

# The optimal threshold (0.276) is lower than the default threshold of 0.5.
# Lowering the threshold makes the model more likely to predict the
# positive class, which increases recall (TPR) and helps capture more
# true positives. However, it also increases false positives.
#
# In a real application, a threshold lower than 0.5 would be chosen when
# missing a positive case is more costly than generating false alarms,
# such as medical screening, fraud detection, or security monitoring.

# ---GridSearch---
# Q1
from sklearn.pipeline import Pipeline

log_reg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=1000))
])

param_grid = {
    "log_reg__C": [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ]
}

grid_search = GridSearchCV(
    log_reg_pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid_search.fit(X_train, y_train)

print("Best C:", grid_search.best_params_["log_reg__C"])

print("Best CV AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_test_probs = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_test_probs)

print("Test AUC:", test_auc)

# GridSearchCV selected C=100.0, which is different from the default
# C=1.0. The grid search improved the cross-validation AUC, but the
# test AUC remained almost unchanged (0.7060 before tuning vs.
# 0.7057 after tuning). This suggests that tuning C did not improve
# generalization performance on the test set.

# Q2

from sklearn.tree import DecisionTreeClassifier

tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier(random_state=42))
])

param_grid_tree = {
    "tree__max_depth": [2, 3, 5, 8, None]
}
tree_grid = GridSearchCV(
    tree_pipeline,
    param_grid_tree,
    cv=5,
    scoring="roc_auc"
)

tree_grid.fit(X_train, y_train)

print("Best max_depth:", tree_grid.best_params_["tree__max_depth"])

print("Best CV AUC:", tree_grid.best_score_)

best_tree = tree_grid.best_estimator_
tree_probs = best_tree.predict_proba(X_test)[:, 1]

tree_test_auc = roc_auc_score(y_test, tree_probs)

print("Test AUC:", tree_test_auc)

# The Decision Tree achieved a higher test AUC (0.9354) compared with
# Logistic Regression (0.7057). The Decision Tree also had a higher
# cross-validation AUC, suggesting it generalizes better on this dataset.
#
# I would bring the Decision Tree into further development because it
# provides substantially better classification performance based on AUC.
# However, AUC is not the only factor to consider. I would also evaluate
# precision, recall, false positive and false negative costs, model
# interpretability, training time, and how stable the model is with
# new data before selecting a final model.

# Q3
tree_grid.cv_results_

cv_results = pd.DataFrame(tree_grid.cv_results_)

cv_summary = cv_results[
    [
        "param_tree__max_depth",
        "mean_test_score",
        "std_test_score"
    ]
]

cv_summary = cv_summary.sort_values(
    by="mean_test_score",
    ascending=False
)

print(cv_summary)

# max_depth=5 achieved the highest mean CV AUC (0.9165) with a
# standard deviation of 0.0213. max_depth=3 had a slightly lower
# mean AUC (0.9024) but a similar and slightly lower standard
# deviation (0.0191), meaning it was slightly more consistent
# across the CV folds.
#
# I would choose max_depth=5 because the improvement in mean AUC
# is meaningful while the increase in standard deviation is small.
# It provides better overall performance without a large loss
# in stability.

# ---joblib---
# Q1
import joblib
import os

best_lr_pipe = grid_search.best_estimator_

os.makedirs("models", exist_ok=True)
joblib.dump(
    best_lr_pipe,
    "assignments_04/models/warmup_model.pkl"
)

loaded_clf = joblib.load(
    "assignments_04/models/warmup_model.pkl"
)

original_preds = best_lr_pipe.predict(X_test)

loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"

print("Predictions match. Model saved and loaded successfully.")

# If only the Logistic Regression model was saved without the scaler,
# the loaded model would receive unscaled features during prediction.
# This would cause incorrect predictions because the model was trained
# using scaled features. The scaler is part of the preprocessing
# workflow and must be saved together with the model.

# Q2

# --- Simulated prediction script ---

import numpy as np
import joblib

# Load the saved pipeline from disk
loaded_model = joblib.load(
    "assignments_04/models/warmup_model.pkl"
)
loaded_model.predict(X_test)

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])


# Predict classes
predicted_classes = loaded_model.predict(new_samples)

# Predict probabilities
predicted_probs = loaded_model.predict_proba(new_samples)[:, 1]


# Print results
for i in range(len(new_samples)):
    print(f"Sample {i+1}")
    print("Predicted class:", predicted_classes[i])
    print("Probability of class 1:", predicted_probs[i])
    print()

# The all-zeros row is predicted as class 1 with a probability of about
# 0.65. Since all features are zero, the prediction is mainly influenced
# by the Logistic Regression intercept and the effect of scaling. The
# model learned that a sample with average/neutral feature values is
# more likely to belong to class 1 in this dataset.