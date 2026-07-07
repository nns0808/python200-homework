import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import flow, task, get_run_logger
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

# File paths
years = list(range(2015, 2025))
BASE_PATH = "../python-200/assignments/resources/happiness_project"
file_paths = [
    f"{BASE_PATH}/world_happiness_{year}.csv"
    for year in years
]

# TASK 1: Load CSV files

@task(retries=3, retry_delay_seconds=2)
def load_file(path):
    logger = get_run_logger()

    df = pd.read_csv(
        path,
        sep=";",
        decimal=","
    )

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 2024 dataset uses ladder_score instead of happiness_score
    if "ladder_score" in df.columns:
        df = df.rename(columns={"ladder_score": "happiness_score"})

    # Add year column
    year = int(os.path.basename(path).split("_")[-1].split(".")[0])
    df["year"] = year

    logger.info(f"Loaded {os.path.basename(path)}")

    return df

#  Combine data
@task
def combine_data(dfs):
    logger = get_run_logger()

    combined = pd.concat(dfs, ignore_index=True)

    logger.info(f"Combined {len(dfs)} files")

    return combined


# TASK 2: Descriptive Statistics

@task
def descriptive_statistics(df):

    logger = get_run_logger()

    # Ensure numeric columns
    df["happiness_score"] = pd.to_numeric(
        df["happiness_score"],
        errors="coerce"
    )

    df["gdp_per_capita"] = pd.to_numeric(
        df["gdp_per_capita"],
        errors="coerce"
    )

    # Overall statistics
    mean_value = df["happiness_score"].mean()
    median_value = df["happiness_score"].median()
    std_value = df["happiness_score"].std()

    logger.info("===== Overall Descriptive Statistics =====")
    logger.info(f"Mean: {mean_value:.3f}")
    logger.info(f"Median: {median_value:.3f}")
    logger.info(f"Standard deviation: {std_value:.3f}")

    # Mean by year
    logger.info("===== Mean Happiness Score by Year =====")

    mean_by_year = (
        df.groupby("year")["happiness_score"]
        .mean()
        .reset_index()
    )

    for _, row in mean_by_year.iterrows():
        logger.info(f"{int(row['year'])}: {row['happiness_score']:.3f}")

    # Mean by region
    logger.info("===== Mean Happiness Score by Region =====")

    mean_by_region = (
        df.groupby("regional_indicator")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
    )

    for region, score in mean_by_region.items():
        logger.info(f"{region}: {score:.3f}")

    return df


# TASK 3: Visual Exploration

@task
def create_visualizations(df):

    logger = get_run_logger()

    output_dir = "assignments_01/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # numeric columns
    df["happiness_score"] = pd.to_numeric(
        df["happiness_score"],
        errors="coerce"
    )

    df["gdp_per_capita"] = pd.to_numeric(
        df["gdp_per_capita"],
        errors="coerce"
    )

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["happiness_score"].dropna(),
        bins=30,
        kde=True
    )
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/happiness_histogram.png")
    plt.close()

    logger.info("Saved happiness_histogram.png")

    # Boxplot by year
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="year",
        y="happiness_score"
    )
    plt.title("Happiness Score by Year")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/happiness_by_year.png")
    plt.close()

    logger.info("Saved happiness_by_year.png")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="gdp_per_capita",
        y="happiness_score"
    )
    plt.title("GDP per Capita vs Happiness Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gdp_vs_happiness.png")
    plt.close()

    logger.info("Saved gdp_vs_happiness.png")

    # Correlation heatmap
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(method="pearson"),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    logger.info("Saved correlation_heatmap.png")



# Task 4: Hypothesis Testing

@task
def hypothesis_testing(df):
    logger = get_run_logger()

    df_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    df_2020 = df[df["year"] == 2020]["happiness_score"].dropna()

    t_stat, p_val = ttest_ind(df_2019, df_2020, equal_var=False)

    mean_2019 = df_2019.mean()
    mean_2020 = df_2020.mean()

    logger.info("===== Hypothesis Test: 2019 vs 2020 =====")
    logger.info(f"2019 mean: {mean_2019:.3f}")
    logger.info(f"2020 mean: {mean_2020:.3f}")
    logger.info(f"t-statistic: {t_stat:.3f}")
    logger.info(f"p-value: {p_val:.5f}")

    alpha = 0.05

    if p_val < alpha:
        logger.info(
            "Result: Significant difference between 2019 and 2020. "
            "This suggests COVID period may have influenced global happiness scores."
        )
    else:
        logger.info(
            "Result: No statistically significant difference between 2019 and 2020. "
            "No measurable global change detected in happiness scores."
        )

  # second test: Europe vs other regions
    if "regional_indicator" in df.columns:
        europe = df[df["regional_indicator"].str.contains("Europe", na=False)]["happiness_score"].dropna()
        others = df[~df["regional_indicator"].str.contains("Europe", na=False)]["happiness_score"].dropna()

        t2, p2 = ttest_ind(europe, others, equal_var=False)

        logger.info("=== Europe vs Non-Europe ===")
        logger.info(f"Europe mean: {europe.mean():.3f}")
        logger.info(f"Others mean: {others.mean():.3f}")
        logger.info(f"t-stat: {t2:.3f}")
        logger.info(f"p-value: {p2:.5f}")

    return df

# Task 5:  Correlation and Multiple Comparisons

@task
def correlation_analysis(df):
    logger = get_run_logger()

    import numpy as np
    from scipy.stats import pearsonr

    target = "happiness_score"

    # numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = [c for c in numeric_cols if c != target]

    results = []

    for col in features:
        temp = df[[col, target]].dropna()

        # skip if constant or too small
        if temp[col].nunique() < 2:
            continue

        r, p = pearsonr(temp[col], temp[target])

        results.append({
            "variable": col,
            "r": r,
            "p_value": p,
            "significant_0.05": p < 0.05
        })

    results_df = pd.DataFrame(results)

    number_of_tests = len(results_df)
    adjusted_alpha = 0.05 / number_of_tests

    results_df["significant_bonferroni"] = (
        results_df["p_value"] < adjusted_alpha
    )

    logger.info("===== Correlation Analysis =====")
    logger.info(f"Number of tests: {number_of_tests}")
    logger.info(f"Bonferroni alpha: {adjusted_alpha:.6f}")

    sig_raw = results_df[results_df["significant_0.05"]]
    sig_bonf = results_df[results_df["significant_bonferroni"]]

    logger.info(f"Significant at 0.05: {len(sig_raw)}")
    logger.info(f"Significant after correction: {len(sig_bonf)}")

    return df

# Task 6: Summary Report
@task
def summary_report(df):
    logger = get_run_logger()

    import numpy as np

    logger.info("===== SUMMARY REPORT =====")

    # 1. Dataset size
    n_countries = df["country_name"].nunique() if "country_name" in df.columns else "Unknown"
    n_years = df["year"].nunique()

    logger.info(f"Total countries: {n_countries}")
    logger.info(f"Total years: {n_years}")

    # 2. Top/bottom regions
    region_means = df.groupby("regional_indicator")["happiness_score"].mean().sort_values()

    top_3 = region_means.tail(3)
    bottom_3 = region_means.head(3)

    logger.info("Top 3 happiest regions:")
    for region, val in top_3.items():
        logger.info(f"{region}: {val:.3f}")

    logger.info("Bottom 3 happiest regions:")
    for region, val in bottom_3.items():
        logger.info(f"{region}: {val:.3f}")

    # 3. T-test interpretation 
    df_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    df_2020 = df[df["year"] == 2020]["happiness_score"].dropna()

    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(df_2019, df_2020, equal_var=False)

    if p_val < 0.05:
        ttest_result = "There WAS a statistically significant difference between 2019 and 2020 happiness scores."
    else:
        ttest_result = "There was NO statistically significant difference between 2019 and 2020 happiness scores."

    logger.info(f"T-test result (2019 vs 2020): {ttest_result}")

    # 4. Most strongly correlated variable 
    numeric_cols = df.select_dtypes(include=np.number).columns
    target = "happiness_score"

    corr_results = []

    for col in numeric_cols:
        if col == target:
            continue

        temp = df[[col, target]].dropna()

        if temp[col].nunique() < 2:
            continue

        r, p = pearsonr(temp[col], temp[target])
        corr_results.append((col, r, p))

    corr_df = pd.DataFrame(corr_results, columns=["variable", "r", "p"])

    if len(corr_df) > 0:
        corr_df["bonferroni"] = corr_df["p"] < (0.05 / len(corr_df))

        sig = corr_df[corr_df["bonferroni"]]

        if len(sig) > 0:
            best = sig.loc[sig["r"].abs().idxmax()]
            logger.info(
                f"Strongest correlation after Bonferroni: {best['variable']} (r={best['r']:.3f})"
            )
        else:
            logger.info("No variables remained significant after Bonferroni correction.")

            return df


@flow
def happiness_pipeline():
    logger = get_run_logger()
    logger.info("Starting Happiness Pipeline")

    dfs = [load_file(path) for path in file_paths]
    combined = combine_data(dfs)

    combined = descriptive_statistics(combined)
    create_visualizations(combined)
 
    combined = hypothesis_testing(combined)
    combined = correlation_analysis(combined)
    
    summary_report(combined)

    logger.info("Pipeline completed successfully.")
    
    return combined

if __name__ == "__main__":
    happiness_pipeline()