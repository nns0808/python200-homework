import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Task 1

# Observation: The CSV file uses semicolons (;) as field separators,
# so pd.read_csv() requires sep=";".

# Load the dataset
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")
os.makedirs("assignments_02/outputs", exist_ok=True)
print(df.columns.tolist())
# Print basic information
print("Shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Plot histogram of final grades (G3)
plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21, edgecolor="black")

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.close()

# Task 2
# Print original shape
print("\nOriginal shape:", df.shape)

# Remove students with G3 = 0
# Reason: A grade of 0 usually indicates the student did not take the final exam.
# These are not true measures of academic performance and would bias the model.
# Students with G3 = 0 generally did not take the final exam.
# Their zero grades are not caused by poor academic performance,
# so including them would distort the regression model.

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


# Removing G3 = 0 students changes the correlation because many of these
# students had high numbers of absences and received a zero simply because
# they missed the final exam. After filtering them out, the remaining data
# better reflects the relationship between attendance and academic performance.
# Compute Pearson correlation between absences and G3

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_filtered["absences"].corr(df_filtered["G3"])

print("\nCorrelation between absences and G3")
print(f"Original dataset : {corr_original:.3f}")
print(f"Filtered dataset : {corr_filtered:.3f}")

# Task 3

# Compute correlations between numeric features and G3
correlations = df_filtered.corr(numeric_only=True)["G3"].sort_values()

print("\nCorrelations with G3:")
print(correlations)

print("\nStrongest positive correlation:")
print(correlations.tail(1))

print("\nStrongest negative correlation:")
print(correlations.head(1))

# Visualization 1: Study Time vs Final Grade
plt.figure(figsize=(6, 4))
plt.scatter(df_filtered["studytime"], df_filtered["G3"], alpha=0.6)

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")

plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.close()

# Students who study more tend to earn slightly higher grades,
# although there is considerable variation among individuals.

# Visualization 2: Absences vs Final Grade

plt.figure(figsize=(6, 4))
plt.scatter(df_filtered["absences"], df_filtered["G3"], alpha=0.6)

plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")

plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.close()

# Students with more absences generally tend to receive lower grades,
# although the relationship is moderate rather than perfect.

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

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade (G3)")
plt.ylabel("Actual Grade (G3)")

plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
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


# Summary:
#
# The filtered dataset contains 357 students after removing students with
# G3 = 0. With an 80/20 train-test split, the test set contains about
# 72 students.
#
# The full model achieved an RMSE of approximately 2.86 grade points and
# a test R² of approximately 0.15. Since grades are measured on a 0-20
# scale, the model's predictions are typically off by about 3 points.
# The R² means that the model explains about 15% of the variation in
# final grades, so many other factors influence student performance.
#
# The largest positive coefficient was internet (+0.834), meaning that
# after accounting for the other features, students with internet access
# were predicted to have slightly higher grades.
#
# The largest negative coefficient was schoolsup (-2.062), meaning students
# receiving school support had lower predicted grades. This was surprising,
# but it may be because students who need additional support are already
# struggling academically, so the variable may indicate higher risk rather
# than causing lower grades.

# Neglected Feature: The Power of G1
# Add G1 to the full model

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
# are already very similar to final grades. However, a high R² does not
# mean that G1 causes G3. It only means that G1 is strongly associated
# with the final grade.
#
# This model is useful for predicting final performance after the first
# grading period, but it is less useful for early intervention because
# educators want to identify struggling students before G1 is available.
#
# To intervene earlier, educators would need to use earlier indicators
# such as attendance, previous failures, study habits, family support,
# engagement, and other factors available before the first-period grade.
# The goal would be to identify risk factors early and provide support
# before students fall behind.