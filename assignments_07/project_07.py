from pathlib import Path
import pandas as pd
from smolagents import tool
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

# Shared DataFrame used by the tools
df = None

# ----Task 1: Define Your Tools----

#  Tool 1: load_happiness_data

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged CSV if it exists. If it does not exist,
    loads and merges the yearly World Happiness CSV files.
    Returns the shape and column names of the loaded dataset.
    """
    global df

    data_path = Path(DATA_PATH)

    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        resources_path = Path(
            "assignments/resources/happiness_project"
        )

        yearly_files = sorted(resources_path.glob("*.csv"))

        if not yearly_files:
            return {"error": "No World Happiness CSV files were found."}

        dataframes = []

        for file in yearly_files:
            yearly_df = pd.read_csv(file)
            dataframes.append(yearly_df)

        df = pd.concat(dataframes, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }

# Tool 2: summarize_column
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary containing descriptive statistics, or an error message.
    """
    if df is None:
        return {"error": "Data has not been loaded. Run load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found in the dataset."}

    return df[column].describe().to_dict()

    
# Tool 3: compute_correlation

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name of the first numeric column.
        col2: The name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson correlation
        coefficient, and p-value, or an error message.
    """
    if df is None:
        return {"error": "Data has not been loaded. Run load_happiness_data first."}

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' was not found in the dataset."}

    if col2 not in df.columns:
        return {"error": f"Column '{col2}' was not found in the dataset."}

    if not pd.api.types.is_numeric_dtype(df[col1]):
        return {"error": f"Column '{col1}' is not numeric."}

    if not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": f"Column '{col2}' is not numeric."}

    try:
        data = df[[col1, col2]].dropna()
        r, p = pearsonr(data[col1], data[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p, 4)
        }

    except Exception as e:
        return {"error": str(e)}

# Tool 4: get_top_n_countries

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a column for a specific year.

    Args:
        column: The numeric column used to rank the countries.
        year: The year to filter the dataset by.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A dictionary containing the year, ranking column, and a list of
        the top countries with their corresponding values, or an error
        message for invalid input.
    """
    if df is None:
        return {
            "error": "Data has not been loaded. Run load_happiness_data first."
        }

    if "year" not in df.columns:
        return {"error": "The dataset does not contain a 'year' column."}

    if "country" not in df.columns:
        return {"error": "The dataset does not contain a 'country' column."}

    if column not in df.columns:
        return {
            "error": f"Column '{column}' was not found in the dataset."
        }

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {
            "error": f"Column '{column}' is not numeric."
        }

    if not isinstance(n, int) or n <= 0:
        return {
            "error": "n must be a positive integer."
        }

    year_data = df[df["year"] == year]

    if year_data.empty:
        return {
            "error": f"No data found for year {year}."
        }

    top_n = (
        year_data
        .sort_values(by=column, ascending=False)
        .head(n)
    )

    results = top_n[["country", column]].to_dict(orient="records")

    return {
        "year": year,
        "column": column,
        "results": results
    }

# ----Task 2: Build the Agent----

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for loading data, summarizing columns, computing
correlations, and ranking countries.

The load_happiness_data tool returns dataset metadata (shape and columns).
When a custom analysis or plot requires the actual rows, read the actual
dataset from:
assignments_01/outputs/merged_happiness.csv

Write Python code directly when the tools are not sufficient, especially
for custom plots using matplotlib.

For plots, always use the actual World Happiness data. Never create,
invent, or use mock data.

Save any plots to assignments_07/outputs/.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=12,
)

# ----Task 3: Run Guided Queries----

if __name__ == "__main__":

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

# ----Task 4: Your Own Questions----

    # My query 1
    my_query_1 = (
        "What is the correlation between healthy_life_expectancy "
        "and happiness_score? Is it statistically significant?"
    )
    print(f"\n--- Custom Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)

    # Comment: This triggered tool use only. The agent used compute_correlation
    # to calculate the Pearson correlation and p-value.

    # My query 2
    my_query_2 = "Show me the top 5 happiest countries in 2024."
    print(f"\n--- Custom Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)

    # Comment: This triggered tool use only. The agent used get_top_n_countries
    # to retrieve the top 5 countries for 2024.


# --- Task 5: Reflection ---
#
# 1. In Query 3, the agent used the p-value to determine whether the correlation was
#    statistically significant. The p-value was 0.0, which is below the common
#    significance threshold of 0.05, so the agent correctly reported that the
#    correlation was statistically significant.
#
# 2. I was surprised by how well the agent could use the available tools to answer
#    different types of questions. For example, it correctly calculated the correlation
#    between healthy_life_expectancy and happiness_score and determined that it was
#    statistically significant. It also used the data to identify the top 5 happiest
#    countries for a specific year.
#
# 3. One additional useful tool would be a tool for filtering and comparing countries
#    based on multiple criteria. It could answer questions such as which countries
#    have both above-average happiness scores and healthy life expectancy, or compare
#    several regions based on selected happiness factors. This would make the agent
#    more useful for more complex data analysis questions.