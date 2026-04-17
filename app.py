import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
import os
import ast


# ==========================================
# 1. SETUP & CONNECTIONS
# ==========================================
st.set_page_config(page_title="Local CFO", page_icon="💰")
st.title("🤖 Local CFO Assistant")

# Connect to Local LLM
llm = ChatOllama(
    model="qwen2.5:1.5b", 
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0 # Keep at 0 for strict SQL generation
)

# Connect to Database (Using the 3-slash rule for local directories)
db = SQLDatabase.from_uri("sqlite:////data/finance.db")

# Setup the Strict SQL Prompt
sql_prompt = PromptTemplate.from_template(
    """You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run.
    Unless otherwise specified, do not return more than {top_k} results.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    
    CRITICAL:
    Respond ONLY with the raw SQL query. Do not include markdown formatting (like ```sql). 
    
    Only use the following tables:
    {table_info}
    
    Question: {input}
    SQL Query:"""
)

# Build the LangChain SQL Chain


# ==========================================
# 2. STREAMLIT UI & MEMORY
# ==========================================
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am connected to your finance database. Ask me anything about your spending!"}
    ]

# Render previous messages on screen
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 3. THE CHAT LOGIC (Formerly ask_cfo)
# ==========================================
user_input = st.chat_input("E.g., How much did I spend on Groceries?")

if user_input:
    # 1. Draw the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 2. Process the AI's response
    with st.chat_message("assistant"):
        with st.spinner("Checking..."):
            try:
                # Generate SQL
                chain = create_sql_query_chain(llm, db, prompt=sql_prompt)
                raw_sql = chain.invoke({"question": user_input})
                clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
                
                # Render the SQL code nicely in the UI so you can verify it
                # st.markdown("**Generated SQL:**")
                # st.code(clean_sql, language="sql")
                
                # Execute against the database
                raw_result = db.run(clean_sql)
                
                # Formatter / Beautifier
                try:
                    parsed_list = ast.literal_eval(raw_result)
                    answer = parsed_list[0][0]
                    if isinstance(answer, (int, float)):
                        final_output = f"💰 **Result:** Rs.{answer:,.2f}"
                    else:
                        final_output = f"💰 **Result:** {answer}"
                except (ValueError, SyntaxError, IndexError):
                    final_output = f"💰 **Result:** {raw_result}"
                
                # Display the final answer
                st.markdown(final_output)
                
                # 3. Save the combined output to memory so it stays on screen
                st.session_state.chat_history.append({"role": "assistant", "content": final_output})

            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})