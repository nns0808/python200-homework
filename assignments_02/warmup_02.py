# --- scikit-learn API --- 
# Q1

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
# Create the model
model = LinearRegression()

# Fit the model
model.fit(years, salary)

# Make predictions
salary_4_years = model.predict([[4]])[0]
salary_8_years = model.predict([[8]])[0]

# Print results
print(f"Slope (coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Predicted salary for 4 years of experience: ${salary_4_years:.2f}")
print(f"Predicted salary for 8 years of experience: ${salary_8_years:.2f}")

# Q2
x = np.array([10, 20, 30, 40, 50])

# Print original shape
print("Original shape:", x.shape)

# Reshape to a 2D array
x = x.reshape(-1, 1)

# Print new shape
print("New shape:", x.shape)

# scikit-learn expects X to be 2D because each row represents one sample
# and each column represents one feature. Even with only one feature,
# the data must have one column so scikit-learn knows the input format.

# Q3

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create a synthetic dataset
X_clusters, _ = make_blobs(
    n_samples=120,
    centers=3,
    cluster_std=0.8,
    random_state=7
)

# Create the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(X_clusters)

# Predict the cluster labels
labels = kmeans.predict(X_clusters)

# Print the cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Print the number of points in each cluster
print("\nPoints in each cluster:")
print(np.bincount(labels))

# Create the scatter plot
plt.figure(figsize=(8, 6))

# Plot the data points colored by cluster
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)

# Plot the cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    c="black",
    label="Cluster Centers"
)

# Add title and labels
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Save the figure
plt.savefig(OUTPUT_DIR / "kmeans_clusters.png")

# Display the plot
plt.show()

# Close the figure
plt.close()

# --- Linear Regression ---


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1

# Create scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    age,
    cost,
    c=smoker,
    cmap="coolwarm"
)

# Add title and labels
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost")

# Save figure
plt.savefig(OUTPUT_DIR / "cost_vs_age.png")

# Display plot
plt.show()

# Close figure
plt.close()

# Comment:
# The plot shows two groups of points: smokers and non-smokers.
# Smokers tend to have higher medical costs because the smoker group
# is shifted upward compared with non-smokers. This suggests that
# smoker status is an important variable that influences medical cost.

# Q2

# Reshape age into a 2D feature array
X = age.reshape(-1, 1)
y = cost
# Split the data into training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
# Create the Linear Regression model
model = LinearRegression()

# Fit the model using training data
model.fit(X_train, y_train)

# Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Calculate R² score
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R²:", r2)

# Comment:
# The slope represents the average change in medical cost for each additional
# year of age. For example, if the slope is about 200, it means that medical
# costs increase by approximately $200 for each additional year of age,
# according to this model.

# Q4

X_full = np.column_stack([age, smoker])
# Split into training and test sets
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    cost,
    test_size=0.2,
    random_state=42
)

# Create and fit the model
model_full = LinearRegression()

model_full.fit(X_train_full, y_train_full)

# Calculate test R²
r2_full = model_full.score(X_test_full, y_test_full)

print("Test R² with age + smoker:", r2_full)

# Compare with Question3 R²
print("Test R² with age only:", r2)

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Q5
# Predict on the test set using the two-feature model
y_pred_full = model_full.predict(X_test_full)

# Create plot
plt.figure(figsize=(8, 6))

# Plot predicted vs actual values
plt.scatter(
    y_pred_full,
    y_test_full
)

# Add diagonal reference line (perfect predictions)
plt.plot(
    [y_test_full.min(), y_test_full.max()],
    [y_test_full.min(), y_test_full.max()]
)

# Add title and labels
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Cost")
plt.ylabel("Actual Medical Cost")

# Save figure
plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png")

# Display plot
plt.show()

# Close figure
plt.close()

# Points above the diagonal line represent cases where the actual medical
# cost was higher than the predicted cost, meaning the model underestimated.
# Points below the diagonal line represent cases where the actual medical
# cost was lower than the predicted cost, meaning the model overestimated.