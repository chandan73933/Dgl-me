from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import psycopg2
import re
import matplotlib.pyplot as plt
import pandas as pd

# PostgreSQL connection details
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'DGliger#3',
    'host': 'database-2.cr48om66eml4.us-east-1.rds.amazonaws.com',
    'port': '5432',
}

# Function to fetch schema details dynamically from the database
def fetch_schema_details():
    schema_info = {}
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Fetch table and column details from the database
        cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """)
        rows = cursor.fetchall()

        # Organize schema details by table
        for table, column, dtype in rows:
            if table not in schema_info:
                schema_info[table] = {}
            schema_info[table][column] = dtype

        cursor.close()
        conn.close()
        return schema_info
    except Exception as e:
        return f"Error fetching schema: {e}"

# Function to generate the schema for the prompt
def generate_schema_prompt(schema_info):
    schema_prompt = "Tables and Columns:\n"
    for table, columns in schema_info.items():
        schema_prompt += f"{table}:\n"
        for column, dtype in columns.items():
            schema_prompt += f"  - {column} ({dtype})\n"
    return schema_prompt

# Updated prompt template to avoid using aliases unless required
template = """[INST]
You are a SQL generation assistant for a PostgreSQL database. Below is the schema information for the database:

{schema}

Make sure to use the correct table names and columns from the schema. Avoid using table aliases unless the query explicitly requires them or to resolve name conflicts. 
When performing JOIN operations, reference the tables directly instead of using aliases unless absolutely necessary.

Generate only the SQL query. Do not provide any additional explanation or text. If you cannot generate a valid SQL query, respond with "INVALID SQL".
User Query: {{question}}
[/INST]
"""

# Function to create the LLM chain
def get_llm_chain(schema_prompt):
    ollama_llm = Ollama(model="llama3.1:8b")
    prompt = PromptTemplate.from_template(template.format(schema=schema_prompt))

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    llm_chain = LLMChain(
        llm=ollama_llm,
        prompt=prompt,
        memory=memory,
        output_key='result'  # Ensure the output key is consistent
    )
    return llm_chain

# Function to execute SQL query
def execute_sql_query(query):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        # Check if the results are not empty
        if results:
            return results
        else:
            return "The query executed successfully, but no data was returned."
    except Exception as e:
        return f"Error executing query: {e}"

# Function to replace aliases with full table names
def replace_aliases_with_table_names(query, schema_info):
    alias_mapping = {}
    
    # Find all alias definitions (e.g., "FROM table_name AS alias")
    alias_pattern = r"FROM\s+(\w+)\s+AS\s+(\w+)"
    matches = re.findall(alias_pattern, query, re.IGNORECASE)
    
    # Create a mapping of aliases to table names
    for table, alias in matches:
        alias_mapping[alias] = table

    # Replace aliases with full table names in the query
    for alias, table in alias_mapping.items():
        query = re.sub(rf"\b{alias}\.", f"{table}.", query)  # Replace alias references in column names
        query = re.sub(rf"\b{alias}\b", f"{table}", query)  # Replace alias with table name in general

    return query

# Post-process the SQL query to normalize case-insensitive comparisons (optional)
def make_query_case_insensitive(query):
    # Regular expression to find column = value comparisons (case sensitive)
    pattern = r"(\w+)\s*=\s*'([^']+)'"
    
    # Replace each occurrence with LOWER(column) = 'value'
    query = re.sub(pattern, lambda m: f"LOWER({m.group(1)}) = '{m.group(2).lower()}'", query)
    
    return query

# Function to generate charts
def generate_chart(results):
    try:
        # Check if results contain a single value (e.g., sum)
        if isinstance(results[0], tuple) and len(results[0]) == 1:
            # Handle single value results like SUM or COUNT
            data = pd.DataFrame([("Total Charges", results[0][0])], columns=['Category', 'Value'])
        else:
            # Ensure results is a list of tuples with at least two columns
            if not results or len(results[0]) != 2:
                raise ValueError("Results must contain two columns for chart generation.")
            data = pd.DataFrame(results, columns=['Category', 'Value'])

        # Check if the Value column is numeric
        data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
        if data['Value'].isna().any():
            raise ValueError("Value column must contain numeric data.")

        # Generate the chart
        data.plot(kind='bar', x='Category', y='Value', legend=False)
        plt.title("Chart Representation")
        plt.xlabel("Category")
        plt.ylabel("Value")

        # Show the chart in a pop-up
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error generating chart: {e}")

# Process user query
def process_user_query(query, schema_info, chart_type):
    # Generate the schema prompt
    schema_prompt = generate_schema_prompt(schema_info)
    
    llm_chain = get_llm_chain(schema_prompt)
    
    # Run the LLM chain with the user query
    response = llm_chain.run(question=query).strip()

    if response.lower() == "invalid sql":
        return "Unable to process the query into a valid SQL statement."
    
    # Log the generated SQL query
    print(f"Generated SQL Query: {response}")

    # Replace aliases with full table names
    response = replace_aliases_with_table_names(response, schema_info)

    # Apply case-insensitivity to the SQL query (optional)
    response = make_query_case_insensitive(response)

    # Check if the generated query is a SELECT statement
    if "SELECT" in response.upper():
        # Execute the SQL query dynamically
        results = execute_sql_query(response)

        # Check if the result is valid
        if isinstance(results, list):
            generate_chart(results)
            return "Chart generated successfully."
        else:
            return results
    else:
        return "The generated response is not a valid SQL query."

# Main interaction function
def chatbot(query, chart_type='bar'):
    # Fetch schema details dynamically
    schema_info = fetch_schema_details()
    
    if isinstance(schema_info, str):  # Handle error in schema fetching
        return schema_info
    
    result = process_user_query(query, schema_info, chart_type)
    print("User Query:", query)
    print("Response:", result)
    return result

# Example usage
if __name__ == "__main__":
    user_query = "how many unique count of female whose age is 55"
    response = chatbot(user_query, chart_type='pie')
    print("Final Response:", response)
