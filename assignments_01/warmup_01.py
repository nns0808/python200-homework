# --- Pandas --- 
# Pandas Q1

import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# Print the first three rows
print("First three rows:")
print(df.head(3))

# Print the shape
print(f"\nShape: {df.shape}")

# Print the number of rows
print(f"Number of rows: {len(df)}")

# Print the number of columns
print(f"Number of columns: {len(df.columns)}")

# Print the data types
print("\nData types:")
print(df.dtypes)

# Pandas Q2

# Filter students who passed and have a grade above 80
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]

print("\nStudents who passed and have a grade above 80:")
print(filtered_df)

# Pandas Q3
df["grade_curved"] = df["grade"] + 5

print("\nUpdated DataFrame:")
print(df)

# Pandas Q4
# Create a new column with uppercase names using .str accessor

df["name_upper"] = df["name"].str.upper()

# Print only name and name_upper columns
print("\nNames in original and uppercase:")
print(df[["name", "name_upper"]])

# Pandas Q5
# Group by city and compute mean grade
city_means = df.groupby("city")["grade"].mean()

print("\nMean grade by city:")
print(city_means)

# Pandas Q6

# Replace "Austin" with "Houston" in the city column
df["city"] = df["city"].replace("Austin", "Houston")

# Print name and city columns to confirm change
print("\nUpdated city values:")
print(df[["name", "city"]])

# Pandas Q7

# Sort by grade in descending order
df_sorted = df.sort_values(by="grade", ascending=False)

# Print top 3 rows
print("\nTop 3 students by grade:")
print(df_sorted.head(3))


# --- NumPy ---

# NumPy Question 1

import numpy as np

# Create 1D NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Print shape
print("\nArray shape:", arr.shape)

# Print data type
print("Array dtype:", arr.dtype)

# Print number of dimensions
print("Array ndim:", arr.ndim)

# NumPy Question 2

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Print shape (rows, columns)
print("\nArray shape:", arr.shape)

# Print total number of elements
print("Array size:", arr.size)

# NumPy Question 3

# Slice the top-left 2x2 block
sub_arr = arr[:2, :2]

print("\nTop-left 2x2 block:")
print(sub_arr)

# NumPy Question 4

# Create a 3x4 array of zeros
zeros_arr = np.zeros((3, 4))

# Create a 2x5 array of ones
ones_arr = np.ones((2, 5))

print("\n3x4 array of zeros:")
print(zeros_arr)

print("\n2x5 array of ones:")
print(ones_arr)

# NumPy Question 5

# Create array from 0 to 45 with step 5
arr5 = np.arange(0, 50, 5)

print("\nArray:", arr5)

# Shape
print("Shape:", arr5.shape)

# Sum
print("Sum:", arr5.sum())

# Mean
print("Mean:", arr5.mean())

# Standard deviation
print("Standard deviation:", arr5.std())

# NumPy Question 6

# Generate 200 random values from a normal distribution (mean=0, std=1)
rand_arr = np.random.normal(0, 1, 200)

print("\nRandom array mean:", rand_arr.mean())
print("Random array standard deviation:", rand_arr.std())

# --- Matplotlib ---

# Matplotlib Question 1

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)

# Add labels and title
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Matplotlib Question 2

subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

# Create bar plot
plt.bar(subjects, scores)

# Add title and labels
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

plt.show()

# Matplotlib Question 3

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

# First dataset (blue)
plt.scatter(x1, y1, color="blue", label="Dataset 1")

# Second dataset (red)
plt.scatter(x2, y2, color="red", label="Dataset 2")

# Labels and title
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("X values")
plt.ylabel("Y values")

# Legend
plt.legend()

# Show plot
plt.show()

# Matplotlib Question 4

# Left subplot data (from Q1)
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Right subplot data (from Q2)
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

# Create 1 row, 2 columns of subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Left subplot: line plot
ax[0].plot(x, y)
ax[0].set_title("Squares")

# Right subplot: bar plot
ax[1].bar(subjects, scores)
ax[1].set_title("Subject Scores")

plt.tight_layout()
plt.show()

# Descriptive Stats 
# Stats Question 1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
arr = np.array(data)

# Mean
print("\nMean:", arr.mean())

# Median
print("Median:", np.median(arr))

# Variance
print("Variance:", arr.var())

# Standard deviation
print("Standard Deviation:", arr.std())

# Stats Question 2

# Generate 500 random values from a normal distribution
scores = np.random.normal(65, 10, 500)

# Create histogram with 20 bins
plt.hist(scores, bins=20)

# title and axis labels
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

# Display the histogram
plt.show()

# Stats Question 3

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])

plt.title("Score Comparison")
plt.xlabel("Groups")
plt.ylabel("Scores")

plt.show()

# Stats Question 4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Create boxplots
plt.boxplot(
    [normal_data, skewed_data],
    tick_labels=["Normal", "Exponential"]
)

# Add title and labels
plt.title("Distribution Comparison")
plt.xlabel("Distribution")
plt.ylabel("Values")

# Display the plot
plt.show()

# Stats Question 5

import statistics

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

# Data 1
print("\nData 1")
print("Mean:", np.mean(data1))
print("Median:", np.median(data1))
print("Mode:", statistics.mode(data1))

# Data 2
print("\nData 2")
print("Mean:", np.mean(data2))
print("Median:", np.median(data2))
print("Mode:", statistics.mode(data2))

# --- Hypothesis Testing ---

# Hypothesis Question 1

from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Perform an independent samples t-test
t_statistic, p_value = stats.ttest_ind(group_a, group_b)

# Print the results
print("\nHypothesis Question 1")
print("T-statistic:", t_statistic)
print("P-value:", p_value)


# Hypothesis Question 6

alpha = 0.05

if p_value < alpha:
    if t_statistic < 0:
        conclusion = (
            "Group A scored significantly lower than Group B. "
            "This difference is unlikely to be due to random chance."
        )
    else:
        conclusion = (
            "Group A scored significantly higher than Group B. "
            "This difference is unlikely to be due to random chance."
        )
else:
    conclusion = (
        "There is no strong evidence of a meaningful difference between Group A and Group B. "
        "The observed difference could be due to random chance."
    )

print("\nHypothesis Question 6")
print(conclusion)

# Hypothesis Question 2

alpha = 0.05

if p_value < alpha:
    print("\nHypothesis Question 2")
    print("\nThe result is statistically significant (p < 0.05).")
else:
    print("\nHypothesis Question 2")
    print("\nThe result is not statistically significant (p >= 0.05).")

# Hypothesis Question 3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Paired t-test (same subjects measured twice)
t_statistic, p_value = stats.ttest_rel(before, after)

print("\nHypothesis Question 3")
print("T-statistic:", t_statistic)
print("P-value:", p_value)

# Hypothesis Question 4

scores = [72, 68, 75, 70, 69, 74, 71, 73]

# One-sample t-test against population mean = 70
t_statistic, p_value = stats.ttest_1samp(scores, 70)

print("\nHypothesis Question 4")
print("T-statistic:", t_statistic)
print("P-value:", p_value)

# Hypothesis Question 5

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# One-tailed independent t-test (A < B)
t_statistic, p_value = stats.ttest_ind(group_a, group_b, alternative="less")

print("\nHypothesis Question 5")
print("T-statistic:", t_statistic)
print("One-tailed P-value:", p_value)

# Correlation review
# Correlation Question 1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)

print("\nCorrelation Question 1")
print("Correlation matrix:\n", corr_matrix)
print("\nCorrelation coefficient:", corr_matrix[0, 1])

# Expected correlation is 1.0 because y increases perfectly linearly with x.
# Every increase in x results in a proportional increase in y (y = 2x),
# so the relationship is positively correlated.

# Correlation Question 2
from scipy.stats import pearsonr
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

# Compute Pearson correlation and p-value
corr_coef, p_value = pearsonr(x, y)

print("\nCorrelation Question 2")
print("Correlation coefficient:", corr_coef)
print("P-value:", p_value)

# Correlation Question 3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}

df = pd.DataFrame(people)

# Compute correlation matrix
corr_matrix = df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# Correlation Question 4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

# Create scatter plot
plt.scatter(x, y)

# Add title and labels
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Correlation Question 5

import seaborn as sns

# Create the heatmap
sns.heatmap(corr_matrix, annot=True)

# Add a title
plt.title("Correlation Heatmap")

plt.show()

# Pipeline

# Pipeline Question 1

arr = np.array([
    12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
    18.0, 14.0, 16.0, 22.0, np.nan, 13.0
])

# Convert NumPy array to pandas Series
def create_series(arr):
    return pd.Series(arr, name="values")

# Remove missing values
def clean_data(series):
    return series.dropna()

# Calculate summary statistics
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Build the pipeline
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

# Run the pipeline
result = data_pipeline(arr)

print("\nPipeline Question 1")

for key, value in result.items():
    print(f"{key}: {value}")