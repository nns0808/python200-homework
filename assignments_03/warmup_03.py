# ------ warmup_03 ----------
# Q1

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Split the data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Print the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2

# Create a StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data only
scaler.fit(X_train)

# Transform both the training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the mean of each feature in the scaled training set
print("Column means of X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# We fit the scaler only on X_train to prevent information from the test set
# from influencing the training process (avoiding data leakage).

# KNN
# Q1

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the unscaled training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Print the accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Q2

knn_scaled = KNeighborsClassifier(n_neighbors=5)

# Train the model on the scaled training data
knn_scaled.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Print the accuracy score
print("Accuracy (scaled data):", accuracy_score(y_test, y_pred_scaled))

#  Scaling made little or no difference because the Iris features are already
# on similar scales, so no single feature dominates the distance calculations.

# Q3

# 5-fold cross-validation on the training data
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

# Print the results
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

# Q4

# Test different values of k
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k = {k}: Mean CV Accuracy = {cv_scores.mean():.4f}")

# I would choose the k with the highest mean cross-validation accuracy.
# If multiple k values have similar scores, I would choose the larger k
# because it is generally less sensitive to noise and less likely to overfit.

# Classifier Evaluation
# Q1
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")
plt.title("KNN Confusion Matrix")

# Save the figure
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()

# The model most often confuses versicolor and virginica.
# Setosa is classified correctly because it is well separated from the other species.

# The sklearn API: Decision Trees
# Q1
# Create the Decision Tree classifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = tree.predict(X_test)

# Print the accuracy score
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))
# The Decision Tree achieved similar accuracy to KNN on the Iris dataset.

# Scaling should make little or no difference for a Decision Tree because
# it splits data based on feature values rather than distance calculations.

# Logistic Regression
# Q1
# Train logistic regression models with different C values
from sklearn.multiclass import OneVsRestClassifier
# Train logistic regression models with different C values
c_values = [0.01, 1.0, 100]

for c in c_values:
    model = OneVsRestClassifier(
        LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear"
        )
    )

    model.fit(X_train_scaled, y_train)

    # Calculate total coefficient magnitude
    coef_size = sum(
        np.abs(estimator.coef_).sum()
        for estimator in model.estimators_
    )

    print(f"C = {c}: Total coefficient magnitude = {coef_size:.4f}")

# As C increases, the total coefficient magnitude increases because
# the regularization becomes weaker, allowing the model to fit the
# training data more closely.

# PCA
# Q1
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# print the shape of X_digits and images
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

# Create a 1-row subplot showing one example of each digit (0-9)
fig, axes = plt.subplots(1, 10, figsize=(12, 2.5))

for digit in range(10):
    # Find the first image of this digit
    index = np.where(y_digits == digit)[0][0]

    axes[digit].imshow(images[index], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()

# Save the figure
plt.savefig("outputs/sample_digits.png")

plt.show()

# Q2
# Create and fit PCA
pca = PCA()
pca.fit(X_digits)

# Transform the data into principal component scores
scores = pca.transform(X_digits)

# Create a scatter plot using the first two principal components
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap="tab10",
    s=10
)

# Add labels and colorbar
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Projection of digits")
plt.colorbar(scatter, label="Digit")

# Save the figure
plt.savefig("outputs/pca_2d_projection.png")

plt.show()

# Same-digit images generally cluster together in this 2D space,
# although there is some overlap between similar-looking digits.

# Q3
# Calculate the cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)

# Save the figure
plt.savefig("outputs/pca_variance_explained.png")

plt.show()

# About 20–21 principal components are needed to explain
# approximately 80% of the variance in the digits dataset.

# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Number of principal components to use
component_list = [2, 5, 15, 40]

# Create figure:
# 1 row for originals + 4 rows for reconstructions
fig, axes = plt.subplots(
    len(component_list) + 1,
    5,
    figsize=(10, 10)
)

# ----- Original images -----
for col in range(5):
    axes[0, col].imshow(images[col], cmap="gray_r")
    axes[0, col].set_title(f"Digit {y_digits[col]}")
    axes[0, col].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12)

# ----- Reconstructed images -----
for row, n in enumerate(component_list, start=1):
    for col in range(5):
        reconstruction = reconstruct_digit(col, scores, pca, n)

        axes[row, col].imshow(reconstruction, cmap="gray_r")
        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(f"{n} PCs", fontsize=12)

plt.tight_layout()

# Save the figure
plt.savefig("outputs/pca_reconstructions.png")

plt.show()

# Around 15 principal components, the digits become clearly recognizable.
# This is consistent with the cumulative variance curve, which begins to level
# off after the first several principal components.