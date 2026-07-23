# Task1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

column_names = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "spam_label"
]

df = pd.read_csv(
    "assignments_03/spambase/spambase.data",
    header=None,
    names=column_names
)

print(df.head())
print(df.shape)

print(df["spam_label"].value_counts())
print(df["spam_label"].value_counts(normalize=True))

# The dataset is somewhat imbalanced, with more ham emails than spam emails.
# Accuracy alone may be misleading because a model could achieve about 60%
# accuracy by predicting every email as ham.

features = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features:
    ham = df[df["spam_label"] == 0][feature]
    spam = df[df["spam_label"] == 1][feature]

    plt.figure(figsize=(6, 4))

    plt.boxplot(
        [ham, spam],
        tick_labels=["Ham", "Spam"]
    )

    plt.title(f"{feature}: Ham vs Spam")
    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{feature}_boxplot.png")

    plt.show()


# The boxplots show that spam emails generally have higher values for
# word_freq_free, char_freq_!, and capital_run_length_total compared with
# ham emails. The differences are noticeable but not complete separation,
# because there is still overlap between the two classes.

# Task2
# Separate features and target
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# stratify=y keeps the same spam/ham ratio in both training and test sets.
# This prevents one split from having too many spam or ham emails.

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only on training data to prevent test data information leakage
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)

# The scaler is fitted only on X_train because using X_test during fitting
# would allow information from the test set to influence preprocessing.

pca = PCA()

# PCA is fitted only on training data to avoid test-set leakage
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.axhline(
    y=0.90,
    linestyle="--"
)

plt.savefig(OUTPUT_DIR / "pca_variance_explained.png")

plt.close()

n = np.argmax(cumulative_variance >= 0.90) + 1

print(f"Number of PCA components needed to reach 90% cumulative explained variance: {n}")

n = np.argmax(cumulative_variance >= 0.90) + 1

print(
    f"n = {n} components are needed to reach "
    f"{cumulative_variance[n-1]:.4f} cumulative explained variance."
)

X_train_pca = pca.transform(X_train_scaled)[:, :n]

X_test_pca = pca.transform(X_test_scaled)[:, :n]

print("Original features:", X_train.shape[1])
print("PCA features:", X_train_pca.shape[1])

# The selected n is the smallest number of principal components that
# preserves at least 90% of the variance in the training data.

# Task3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Collect model comparison results
results = []

def save_result(name, accuracy):
    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

knn_unscaled = KNeighborsClassifier(n_neighbors=5)

knn_unscaled.fit(X_train, y_train)

y_pred_knn_unscaled = knn_unscaled.predict(X_test)

knn_unscaled_accuracy = accuracy_score(
    y_test,
    y_pred_knn_unscaled
)

print("KNN Unscaled Accuracy:")
print(knn_unscaled_accuracy)

save_result(
    "KNN Unscaled",
    knn_unscaled_accuracy
)

print(classification_report(y_test, y_pred_knn_unscaled))

knn_scaled = KNeighborsClassifier(n_neighbors=5)

knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

knn_scaled_accuracy = accuracy_score(
    y_test,
    y_pred_knn_scaled
)

print("KNN Scaled Accuracy:")
print(knn_scaled_accuracy)

save_result(
    "KNN Scaled",
    knn_scaled_accuracy
)

print(classification_report(y_test, y_pred_knn_scaled))

knn_pca = KNeighborsClassifier(n_neighbors=5)

knn_pca.fit(X_train_pca, y_train)

y_pred_knn_pca = knn_pca.predict(X_test_pca)

knn_pca_accuracy = accuracy_score(
    y_test,
    y_pred_knn_pca
)

print("KNN PCA Accuracy:")
print(knn_pca_accuracy)

save_result(
    "KNN PCA",
    knn_pca_accuracy
)

print(classification_report(y_test, y_pred_knn_pca))

# PCA may improve KNN because reducing dimensions can remove noise and
# make distance calculations more meaningful.

tree_results = []

depths = [3, 5, 10, None]

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)

    tree_results.append({
        "Depth": depth,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy
    })

tree_results_df = pd.DataFrame(tree_results)

print(tree_results_df)

best_depth = 5

print(f"\nProduction choice: Decision Tree (max_depth={best_depth})")

# The table above compares both training and test accuracy for each depth.
# I selected max_depth=5 for production because it provided a good balance
# between training and test accuracy. Deeper trees achieved higher training
# accuracy but showed little or no improvement in test accuracy, indicating
# an increased risk of overfitting.

tree = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("Decision Tree Accuracy:")
print(accuracy_score(y_test, y_pred_tree))

tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

save_result(
    "Decision Tree depth=5",
    tree_accuracy
)

print("\nSelected production model: Decision Tree (max_depth=5)")



# Decision Tree feature importance

tree_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": tree.feature_importances_
})

tree_importance = tree_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Decision Tree Features:")
print(tree_importance.head(10))

# Random forest

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("Random Forest Accuracy:")
print(rf_accuracy)

save_result(
    "Random Forest",
    rf_accuracy
)

print(classification_report(y_test, y_pred_rf))

# Random Forest feature importance

rf_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Random Forest Features:")
print(rf_importance.head(10))

# Compare Decision Tree and Random Forest feature importances

comparison = pd.DataFrame({
    "Feature": X_train.columns,
    "Decision Tree Importance": tree.feature_importances_,
    "Random Forest Importance": rf.feature_importances_
})

comparison = comparison.sort_values(
    by="Random Forest Importance",
    ascending=False
)

print("\nTop 10 Feature Importance Comparison:")
print(comparison.head(10))

# Both the Decision Tree and the Random Forest identified many of the same
# features as important for distinguishing spam from ham, indicating agreement
# on the strongest predictors. Features related to word frequency, special
# characters, and capitalization patterns consistently ranked among the most
# important in both models.

# The Random Forest importance values are generally more reliable because they
# are averaged across many decision trees, making them less sensitive to the
# specific training sample than a single Decision Tree.


# Plot top 10 Random Forest feature importances

top10_rf = rf_importance.head(10)

plt.figure(figsize=(8, 5))

plt.barh(
    top10_rf["Feature"],
    top10_rf["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importances.png"
)

plt.show()

# Logistic regression scaled data

logreg = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

logreg.fit(X_train_scaled, y_train)

y_pred_logreg = logreg.predict(X_test_scaled)

print("Logistic Regression Scaled Accuracy:")
print(accuracy_score(y_test, y_pred_logreg))

logreg_accuracy = accuracy_score(
    y_test,
    y_pred_logreg
)

save_result(
    "Logistic Regression",
    logreg_accuracy
)

print(classification_report(y_test, y_pred_logreg))

# Logistic regression PCA data

logreg_pca = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

logreg_pca.fit(X_train_pca, y_train)

y_pred_logreg_pca = logreg_pca.predict(X_test_pca)

print("Logistic Regression PCA Accuracy:")
print(accuracy_score(y_test, y_pred_logreg_pca))

logreg_pca_accuracy = accuracy_score(
    y_test,
    y_pred_logreg_pca
)

save_result(
    "Logistic Regression PCA",
    logreg_pca_accuracy
)

print(classification_report(y_test, y_pred_logreg_pca))


# Overall, Random Forest/Logistic Regression/KNN achieved the strongest
# performance. PCA did not necessarily improve every model because reducing
# dimensions can remove useful information. However, PCA can help models
# that are sensitive to feature scale and distance calculations.
#
# Accuracy alone is not the best metric for spam filtering. False positives
# are especially costly because legitimate emails may be incorrectly marked
# as spam. A good spam filter should balance precision and recall, depending
# on whether preventing spam or protecting legitimate emails is more important.


# Model comparison summary

comparison = pd.DataFrame(results)

print("\nModel Comparison:")
print(
    comparison.sort_values(
        "Accuracy",
        ascending=False
    )
)
# Confusion matrix


best_model_predictions = y_pred_rf

cm = confusion_matrix(
    y_test,
    best_model_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()

plt.title("Best Model Confusion Matrix")

plt.savefig(
    OUTPUT_DIR / "best_model_confusion_matrix.png"
)

plt.close()

# The confusion matrix shows whether the model makes more false positives
# (ham classified as spam) or false negatives (spam classified as ham).
# For spam filtering, false positives are often more harmful because users
# may lose important legitimate emails.

# Random Forest achieved the highest overall accuracy (94.5%) and
# performed better than the other classifiers. Scaling greatly improved
# KNN performance compared with using the unscaled data. PCA produced
# only a very small improvement for KNN but slightly reduced the accuracy
# of Logistic Regression. This suggests that PCA did not provide a major
# benefit for this dataset because the original features already contain
# useful information.
#
# For a spam filter, accuracy alone is not the most important metric.
# False positives, where legitimate emails are marked as spam, can cause
# users to miss important messages. Therefore, minimizing false positives
# is generally more important than maximizing overall accuracy.

# Task4

# knn unscaled
cv_results = []

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(
    knn,
    X_train,
    y_train,
    cv=5
)

print("KNN (Unscaled)")
print("Fold scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

cv_results.append({
    "Model": "KNN (Unscaled)",
    "Mean Accuracy": scores.mean(),
    "Std Accuracy": scores.std()
})

print()

# knn scaled
knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(
    knn,
    X_train_scaled,
    y_train,
    cv=5
)

print("KNN (Scaled)")
print("Fold scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

cv_results.append({
    "Model": "KNN (Scaled)",
    "Mean Accuracy": scores.mean(),
    "Std Accuracy": scores.std()
})

print()

# knn pca
knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(
    knn,
    X_train_pca,
    y_train,
    cv=5
)

print("KNN (PCA)")
print("Fold scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

cv_results.append({
    "Model": "KNN (PCA)",
    "Mean Accuracy": scores.mean(),
    "Std Accuracy": scores.std()
})

print()

# decision tree

tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

scores = cross_val_score(
    tree,
    X_train,
    y_train,
    cv=5
)

print("Decision Tree")
print("Fold scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

cv_results.append({
    "Model": "Decision Tree",
    "Mean Accuracy": scores.mean(),
    "Std Accuracy": scores.std()
})

print()

# random forest

rf = RandomForestClassifier(random_state=42)

scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=5
)

print("Random Forest")
print("Fold scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

cv_results.append({
    "Model": "Random Forest",
    "Mean Accuracy": scores.mean(),
    "Std Accuracy": scores.std()
})

print()

# logistic regression(scaled)

logreg = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

scores = cross_val_score(
    logreg,
    X_train_scaled,
    y_train,
    cv=5
)

print("Logistic Regression (Scaled)")
print("Fold scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

cv_results.append({
    "Model": "Logistic Regression (Scaled)",
    "Mean Accuracy": scores.mean(),
    "Std Accuracy": scores.std()
})

print()

# logistiv regression(pca)

logreg = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

scores = cross_val_score(
    logreg,
    X_train_pca,
    y_train,
    cv=5
)

print("Logistic Regression (PCA)")
print("Fold scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

cv_results.append({
    "Model": "Logistic Regression (PCA)",
    "Mean Accuracy": scores.mean(),
    "Std Accuracy": scores.std()
})

print()

cv_results_df = pd.DataFrame(cv_results)

print("\nCross Validation Comparison:")
print(
    cv_results_df.sort_values(
        by="Mean Accuracy",
        ascending=False
    )
)

# Five-fold cross-validation was used for all classifiers.
# The comparison includes mean accuracy and standard deviation across folds.
# Random Forest achieved the highest average accuracy, while models with lower
# standard deviation showed more consistent performance across different splits.

# Cross-validation confirmed the ranking observed in the test-set comparison.
# Random Forest achieved the highest mean accuracy and showed consistent
# performance across the five folds, making it the strongest classifier for
# this dataset.
#
# Logistic Regression showed stable performance with a relatively small
# standard deviation, indicating consistent results across different splits.
#
# The Decision Tree depth analysis showed that increasing tree depth improves
# training accuracy but can lead to overfitting when the tree becomes too
# complex. The selected max_depth=5 provided a balance between performance
# and generalization.
#
# Feature importance analysis showed that both Decision Tree and Random Forest
# relied on specific email characteristics, such as word frequency, character
# frequency, and capital letter patterns, to distinguish spam from ham.
#
# Overall, Random Forest provided the best combination of accuracy, stability,
# and ability to capture complex patterns in the Spambase dataset.

# task5
# My pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# random forest pipeline

rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)

rf_pipeline_accuracy = accuracy_score(y_test, y_pred_rf)

print("Random Forest Pipeline")
print("Accuracy:", rf_pipeline_accuracy)
print(classification_report(y_test, y_pred_rf))

# logistic regression pipeline

# logistic regression pipeline

logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ))
])

logreg_pipeline.fit(X_train, y_train)

y_pred_log = logreg_pipeline.predict(X_test)

logreg_pipeline_accuracy = accuracy_score(y_test, y_pred_log)

print("Logistic Regression Pipeline")
print("Accuracy:", logreg_pipeline_accuracy)
print(classification_report(y_test, y_pred_log))


# The Logistic Regression pipeline uses the best non-tree preprocessing
# approach identified in Task 3. Since PCA slightly reduced Logistic
# Regression performance on this dataset, the pipeline uses the
# StandardScaler - Logistic Regression workflow without PCA.

# The pipeline applies StandardScaler before Logistic Regression,
# matching the preprocessing used in the best-performing non-tree model.
# The pipeline prevents data leakage by fitting preprocessing steps only
# on the training data and applying the same transformations to new data.


# ---What is the practical value of packaging a model---

# Pipelines make machine learning workflows easier to use and less
# error-prone. They automatically apply preprocessing in the correct
# order, help prevent data leakage by fitting preprocessing steps only
# on the training data, and allow the entire workflow to be treated as
# a single object. This makes it much easier to share or deploy a model
# because users only need to call fit() and predict() without worrying
# about the preprocessing steps.