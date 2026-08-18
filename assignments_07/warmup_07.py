# ----Lesson 02: Tool Definitions and the ReAct Loop----
# Q1

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dotenv import load_dotenv
from openai import OpenAI
import os
from datetime import datetime
import json

load_dotenv()

client = OpenAI()

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "The temperature in degrees Celsius."
            }
        },
        "required": ["celsius"]
    }
}


print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Q2

from datetime import datetime


def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


get_current_time()


tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]

print('Tools list defined with one tool: get_current_time')


def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
Use the tool get_current_time whenever a user asks about the time.'''

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 1: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    print("First response received from model...")
    print(first_response)

    first_message = first_response.choices[0].message

    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 2: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 3: second API call
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''

    else:
        print("No tools needed....")

    return first_message.content or ''


# Prediction
#
# 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit")
#    trigger a tool call?
#
#    Prediction: No. The only available tool is get_current_time,
#    and converting Celsius to Fahrenheit does not require the
#    current time.
#
# 2. How many API calls will be made to answer this query?
#
#    Prediction: One API call. Since no tool call should be needed,
#    the agent should return the answer after the first API call.

result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Final result:", result)

# Prediction check:
# Yes, the prediction was correct. No tool call was triggered,
# and the model answered the conversion directly.

# Q3

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current local time as a formatted string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": celsius_to_fahrenheit_schema
    }
]


def run_agent(user_prompt: str) -> str:
    """Run the agent with both get_current_time and celsius_to_fahrenheit tools."""

    SYSTEM_PROMPT = """You are a helpful assistant that can tell the current time
    and convert Celsius temperatures to Fahrenheit.
    Use get_current_time whenever a user asks about the time.
    Use celsius_to_fahrenheit whenever a user asks to convert Celsius to Fahrenheit."""

    # Step 1: start the conversation with system and user messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")
    print(first_response)

    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")

        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name

            if function_name == "get_current_time":
                tool_result = get_current_time()

            elif function_name == "celsius_to_fahrenheit":
                arguments = json.loads(tool_call.function.arguments)
                celsius = arguments["celsius"]
                tool_result = celsius_to_fahrenheit(celsius)

            else:
                tool_result = f"Error: unknown tool {function_name}."

            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ""

    else:
        print("No tools needed....")

    return first_message.content or ""


response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A tool was called because celsius_to_fahrenheit is available and
# the question specifically asks for a Celsius-to-Fahrenheit conversion.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# No tool was called because the question can be answered directly
# without either available tool.

# ----Lesson 03: Multi-Tool Agent----
# Q4

# CsvManager

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []

        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"

            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }

        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()

        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }

        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()

        if error:
            return error

        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()

        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]

            if missing:
                return {
                    "error": f"These columns are not in the data: {missing}"
                }

            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)

        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()

        if error:
            return error

        if column not in self.df.columns:
            return {
                "error": (
                    f"'{column}' is not a column. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(
        self,
        y: str,
        x: str | None = None,
        plot_type: str = "line"
    ):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()

        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return (
                f"Error: column '{y}' is not in "
                f"{self.df.columns.tolist()}"
            )

        # If someone accidentally passes x == y,
        # treat it like "plot y"
        if x == y:
            x = None

        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(
                f"{title_csv} | Line plot: {y} vs row index"
            )
            plt.show()

            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return (
                f"Error: column '{x}' is not in "
                f"{self.df.columns.tolist()}"
            )

        ax = self.df.plot(
            x=x,
            y=y,
            kind=plot_type
        )

        ax.set_title(
            f"{title_csv} | "
            f"{plot_type.title()} plot: {y} vs {x}"
        )

        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    # Q4 NEW TOOL
    
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns
        in the loaded DataFrame.

        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()

        if error:
            return error

        if col1 not in self.df.columns:
            return {
                "error": (
                    f"'{col1}' is not a column. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        if col2 not in self.df.columns:
            return {
                "error": (
                    f"'{col2}' is not a column. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        r, p_value = pearsonr(
            self.df[col1],
            self.df[col2]
        )

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p_value), 4),
        }


print("Class defined")


# Create CsvManager

resources_dir = Path(__file__).parent / "resources"

csv_manager = CsvManager(resources_dir)


# Tool schemas


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in resources/.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from resources/ and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The CSV filename to load."
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Return column names for the currently loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Return basic summary statistics for columns in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of columns to summarize."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Return summary statistics for one column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The column to describe."
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Create a line or scatter plot from the active CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description": "The y-axis column."
                    },
                    "x": {
                        "type": "string",
                        "description": "Optional x-axis column."
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter"],
                        "description": "The type of plot."
                    }
                },
                "required": ["y"]
            }
        }
    },

    
    # Q4 NEW SCHEMA
    
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute the Pearson correlation coefficient and "
                "p-value between two columns in the loaded CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "The name of the first column."
                    },
                    "col2": {
                        "type": "string",
                        "description": "The name of the second column."
                    }
                },
                "required": ["col1", "col2"]
            }
        }
    }
]


# Map tool names to Python methods


node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,

    # Q4 NEW TOOL
    "compute_correlation": csv_manager.compute_correlation,
}



# Run agent cycle


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    """

    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        content = (
            json.dumps(result, default=str)
            if not isinstance(result, str)
            else result
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    for loop_idx in range(max_tool_rounds):

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        assistant_entry = {
            "role": "assistant",
            "content": msg.content
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                tc.model_dump() for tc in msg.tool_calls
            ]

        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)

            if fn is None:
                result = {
                    "error": f"Tool '{name}' not found."
                }
            else:
                try:
                    result = (
                        fn(**tool_args)
                        if tool_args
                        else fn()
                    )
                except Exception as e:
                    result = {
                        "error": (
                            f"Tool '{name}' failed: "
                            f"{type(e).__name__}: {e}"
                        )
                    }

            messages.append(
                observe_tool_result(
                    tool_call.id,
                    result
                )
            )

    return "I hit the tool-round limit. Try a simpler request."



# Q4 test

SYSTEM_PROMPT = """
You are a helpful CSV data analysis assistant.

You can use tools to:
- list CSV files
- load a CSV
- inspect columns
- summarize columns
- describe columns
- create plots
- compute Pearson correlations

Use the available tools when they are needed to answer the user's question.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

response = run_agent_cycle(
    messages,
    "Load the bike_commute CSV and compute the correlation "
    "between distance_km and duration_min."
)

print("\nQ4 Response:")
print(response)

# Q5

SYSTEM_PROMPT = """
You are a helpful CSV data analysis assistant.

You can use tools to:
- list CSV files
- load a CSV
- inspect columns
- summarize columns
- describe columns
- create plots
- compute Pearson correlations

Use the available tools when they are needed to answer the user's question.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print(result)

# Q6
# system: gives the agent its instructions and available behavior.
# user: contains the user's request.
# assistant: contains the model's reasoning/tool requests or final response.
# tool: contains the results returned by the Python tools to the model.

import json

print(json.dumps(messages, indent=2, default=str))

# ----Lesson 04: smolagents----
# Q7

from smolagents import tool


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded CSV.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.
    """
    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)

# Smolagents generates the tool description and parameter schema
# automatically from the function name, type hints, and docstring.
# Unlike Q4, where I manually wrote the JSON schema, smolagents
# creates the schema automatically. The developer needs to provide
# a clear function name, type hints, and descriptions for each
# parameter so smolagents can generate a useful tool description.

# Q8

from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns or all columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column using basic statistics.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic statistics for the column.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(
    y: str,
    x: str | None = None,
    plot_type: str = "line"
) -> str | dict:
    """Plot data from the active CSV.

    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        A success message or error.
    """
    return csv_manager.plot_data(
        y=y,
        x=x,
        plot_type=plot_type
    )



TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

model = OpenAIServerModel(
    model_id="gpt-4.1-mini"
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)

response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager}
)

print("ToolCallingAgent response:")
print(response_tool)

print("CodeAgent response:")
print(response_code)

# Comparison:
#
# ToolCallingAgent:
# The ToolCallingAgent loaded the CSV and called the plot_data tool
# to create the requested scatter plot. However, it did NOT actually
# change the dots to green because plot_data does not provide a color
# parameter. The agent's final response incorrectly stated that the
# plot had been created with green dots.
#
# CodeAgent:
# The CodeAgent loaded the CSV and created the scatter plot using
# plot_data. It recognized that the user requested green dots and
# attempted to pass color="green" to plot_data, but this failed because
# plot_data does not accept a color argument. It then attempted to use
# matplotlib directly, but that import was blocked by the CodeAgent's
# authorized-import restrictions. Therefore, the CodeAgent also did
# NOT actually create green dots.
#
# Overall:
# Neither agent actually changed the dots to green. The ToolCallingAgent
# was limited to the parameters exposed by the available tools, so it
# is most useful when the task can be completed using well-defined,
# predefined tools.
#
# The CodeAgent was more flexible because it can generate and execute
# Python code, making it more useful for tasks that require custom
# calculations, transformations, or actions that are not directly
# supported by predefined tools. In this example, however, the
# CodeAgent was also limited by the environment's authorized-import
# restrictions.

# Q9
#
# 1. A ToolCallingAgent would be a better choice for a task where the
# available actions are well-defined and can be handled by a set of
# predefined tools. For example, answering questions about a CSV file
# using tools such as load_csv, get_columns, and compute_correlation
# would be a good fit. The task is well suited to a tool-based approach
# because the agent only needs to select and use specific tools rather
# than generate new Python code.
#
# 2. A meaningful risk of using a CodeAgent is that it generates and
# executes Python code, which can perform actions beyond the specific
# predefined tools. This creates a greater risk of unintended code
# execution or unexpected changes to data or files. A ToolCallingAgent
# is more constrained because it can only call the tools that have been
# explicitly provided to it.