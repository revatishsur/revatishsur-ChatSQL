from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv
import sqlite3
import sqlparse
import json
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

# Ignore specific deprecation warnings from langchain_core
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


load_dotenv()

# Create and initialize the OpenAI client
def create_openai_client():
    return ChatOpenAI(model_name="gpt-3.5-turbo")

# Retrieve the database schema from an SQLite database
def get_database_schema(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_info = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            primary_keys = [col[1] for col in columns if col[5] == 1]
            schema_info.append(f"{table_name} (Primary Key: {', '.join(primary_keys)})")

        conn.close()
        return "filename: chinook.db, "+', '.join(schema_info)
    except Exception as e:
        return f"Error reading database schema: {e}"

# Format the raw SQL query for better readability using sqlparse
def format_sql_query(raw_sql):
    if raw_sql.startswith("```sql"):
        raw_sql = raw_sql[6:]
    if raw_sql.endswith("```"):
        raw_sql = raw_sql[:-3]
    raw_sql = raw_sql.strip()
    formatted_sql = sqlparse.format(raw_sql, reindent=True, keyword_case='upper')
    return formatted_sql

# Function to execute a SQL query and return results
def execute_sql_query(db_path, sql_query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        return f"SQL Error: {e}"


# Main execution
if __name__ == "__main__":
    client = create_openai_client()
    db_path = "chinook.db"
    database_schema = get_database_schema(db_path)

    memory = ConversationBufferMemory(
        chat_memory=FileChatMessageHistory("chat_history.json"),
        memory_key="messages",
        return_messages=True,
    )

    prompt = ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template(
                "You are a data analyst. Generate SQL queries from natural language descriptions. Provide only the SQL query in response, without explanations or additional text"
                "Question: {{question}} Database Schema: {{database_schema}} Response: {content}"
            ),
        ],
    )

    chain = LLMChain(
        llm=client,
        prompt=prompt,
        memory=memory,
    )

    while True:
        content = input("Usage: type 'quit' to exit chat\n>> ")

        if content == "quit":
            break

        result = chain({"content": content})

        # Format the SQL query
        formatted_query = format_sql_query(result['text'])
        print("\nGenerated SQL Query:\n", formatted_query)

        # Execute the SQL query and print results
        query_results = execute_sql_query(db_path, formatted_query)
        print("\nQuery Results:\n", query_results)