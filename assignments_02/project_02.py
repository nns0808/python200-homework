from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "student_performance_math.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Task 1

# Observation: The CSV file uses semicolons (;) as field separators,
# so pd.read_csv() requires sep=";".

# Load the dataset

df = pd.read_csv(DATA_FILE, sep=";")
print(df.columns.tolist())

# Print basic information
print("Shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Task 2: Preprocess the Data
# Plot histogram of final grades (G3)

plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21, edgecolor="black")

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig(OUTPUT_DIR / "g3_distribution.png")
plt.close()

df_filtered = df[df["G3"] > 0].copy()

print("Filtered shape:", df_filtered.shape)
print("Rows removed:", len(df) - len(df_filtered))

yes_no_columns = [
    "schoolsup",
    "internet",
    "higher",
    "activities",
]

for col in yes_no_columns:
    df_filtered[col] = df_filtered[col].map({"yes": 1, "no": 0})

# Convert sex to 0/1
# Female = 0, Male = 1
df_filtered["sex"] = df_filtered["sex"].map({"F": 0, "M": 1})


# Removing students with G3 = 0 changes the correlation because many of
# these students did not actually complete the final exam. Their grade of
# zero reflects missing the exam rather than their true academic performance.
# Including these students mixes together exam non-participation and academic
# achievement, making the relationship between absences and final grade appear
# weaker than it really is. After filtering out the G3 = 0 rows, the remaining
# data better represents students who completed the course, so the correlation
# more accurately reflects how increased absences are associated with lower
# final grades.

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_filtered["absences"].corr(df_filtered["G3"])

print("\nCorrelation between absences and G3")
print(f"Original dataset : {corr_original:.3f}")
print(f"Filtered dataset : {corr_filtered:.3f}")


# Task 3: Exploratory Data Analysis

# Compute Pearson correlations between all numeric features and G3
# using the filtered dataset.

correlations = (
    df_filtered
    .corr(numeric_only=True)["G3"]
    .sort_values()
)

print("\nPearson correlations with G3:")
print(correlations)

print("\nMost positive correlation:")
print(correlations.tail(1))

print("\nMost negative correlation:")
print(correlations.head(1))


# Visualization 1:
# Study time vs final grade

plt.figure(figsize=(6, 4))

plt.scatter(
    df_filtered["studytime"],
    df_filtered["G3"],
    alpha=0.6
)

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")

plt.savefig(OUTPUT_DIR / "studytime_vs_g3.png")

plt.close()


# Comment:
# This scatter plot shows the relationship between study time
# and final grades. Students with higher study time tend to have
# slightly higher G3 scores, although there is substantial
# variation between students.


# Visualization 2:
# Absences vs final grade

plt.figure(figsize=(6, 4))

plt.scatter(
    df_filtered["absences"],
    df_filtered["G3"],
    alpha=0.6
)

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.savefig(OUTPUT_DIR / "absences_vs_g3.png")

plt.close()


# Comment:
# This scatter plot shows that higher absences generally correspond
# to lower final grades. The relationship is not perfect because
# student performance is influenced by many other factors.

# Task 4
# Feature (X) and target (y)
X = df[["failures"]]
y = df["G3"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Calculate R²
r2 = model.score(X_test, y_test)

# Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R²:", r2)

# Comment:
# The slope shows how much the predicted final grade (G3) changes
# for each additional previous failure. A negative slope means that
# students with more failures tend to have lower final grades.
#
# The RMSE shows the average prediction error in grade points.
# Since grades range from 0 to 20, an RMSE of about 2 means the
# predictions are typically off by about 2 grade points.
#
# The R² value tells us how much of the variation in G3 is explained
# by the number of failures alone. Because failures is only one factor
# affecting student performance, we expect R² to be relatively low
# compared with a model that includes additional features such as G1,
# G2, study time, or absences.

# Task 5

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
df_clean = df_filtered
X = df_clean[feature_cols].values
y = df_clean["G3"].values


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("RMSE:", rmse)

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Adding more features improved the model.
# The baseline model explained about 8% of the variation in final grades,
# while the full model explains about 15%. This is an improvement,
# although the model still leaves much of the variation unexplained.

# The training and test R² values are fairly close, which suggests that
# the model is not strongly overfitting. It performs similarly on both
# the training data and unseen test data, although its overall predictive
# power is modest.

# If deploying this model, I would keep failures, studytime, higher,
# schoolsup, internet, and the parents' education variables because
# they have the largest coefficients and appear to contribute the most
# to the predictions. I would consider dropping activities and freetime
# because their coefficients are very close to zero, suggesting they add
# little predictive value. Traveltime also has only a small effect.

# Task 6: Evaluate and Summarize

# Predict using the full model
y_pred = model.predict(X_test)

# Create predicted vs actual plot
plt.figure(figsize=(8, 6))

plt.scatter(y_pred, y_test)

# Add diagonal reference line (perfect predictions)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Grade (G3)")
plt.ylabel("Actual Grade (G3)")

plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png")
plt.close()


# Comment:
# Points above the diagonal line represent students whose actual grade
# was higher than the model predicted (the model underestimated).
# Points below the diagonal line represent students whose actual grade
# was lower than the model predicted (the model overestimated).
#
# The model appears to have errors across the range of grades rather than
# only at the high or low end. The prediction error is relatively uniform,
# although there may be more difficulty predicting extreme grades because
# fewer students have very high or very low scores.


# ==========================
# Summary
# ==========================

# The filtered dataset contains 357 students after removing the
# students with G3 = 0. Using an 80/20 train-test split resulted
# in a test set containing 72 students.

# The full regression model achieved an RMSE of approximately 2.86,
# meaning that the predictions are typically within about 3 grade
# points of the actual final grade.

# The model achieved a test R² of approximately 0.15, so it explains
# about 15% of the variation in students' final grades. This indicates
# that many additional factors beyond the selected features influence
# academic performance.

# In the full model, internet had the largest positive coefficient
# (+0.834), meaning students with internet access were predicted to
# have slightly higher grades after accounting for the other features.

# School support (schoolsup) had the largest negative coefficient
# (-2.062). This does not mean school support lowers grades. Rather,
# students receiving school support are more likely to already be
# struggling academically, so school support is acting as an indicator
# of students who need additional help rather than causing lower grades.

# Overall, the model captures some meaningful relationships between
# student characteristics and final grades, but its predictive power
# is limited because many important influences on academic performance
# are not included in the model.

feature_cols_g1 = [
    "failures", "Medu", "Fedu", "studytime", "higher",
    "schoolsup", "internet", "sex", "freetime",
    "activities", "traveltime", "G1"
]

X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

# Split data

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1,
    y_g1,
    test_size=0.2,
    random_state=42
)

# Fit model

model_g1 = LinearRegression()

model_g1.fit(X_train_g1, y_train_g1)

# Test R²

test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)

print("Test R² with G1:", test_r2_g1)

# Adding G1 greatly improves the model's R² because first-period grades
# are strongly associated with final grades. However, a high R² does not
# mean that G1 causes G3. It only means that G1 is a strong predictor of
# G3. The model identifies a correlation between first-period performance
# and final grades, but it does not prove that improving G1 alone will
# directly cause higher G3 scores.

# G1 is useful for prediction because it summarizes many factors that
# already influence student performance, such as prior knowledge,
# motivation, study habits, attendance, and engagement. However, using
# G1 for early intervention is limited because educators only observe
# this information after the first grading period.

# Before G1 is available, educators would need to rely on earlier warning
# indicators such as previous failures, absences, study time, participation,
# family support, and student engagement. These factors could help identify
# students who may be at risk and allow schools to provide support before
# academic difficulties become reflected in their first-period grades.