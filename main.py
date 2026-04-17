import asyncio
import time
import os
import sqlite3
import pandas as pd
import glob
import shutil
from typing import TypedDict, Sequence, List, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. Define State and Model
# ---------------------------------------------------------
class AgentState(TypedDict):
    current_index: int            # Tracks which row we are on
    is_done: bool                 # Flag to stop the loop
    raw_data: str                 # The current chunk of Markdown
    master_transactions: List[Any] # Accumulates Pydantic objects
    final_report: str             # The final string output


# Connect to Local Mistral via Docker internal host
llm = ChatOllama(
    model="qwen2.5:1.5b", #mistral:7b
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
)

structured_llm = llm.with_structured_output(CategoryMapping)

# ---------------------------------------------------------
# 3. Define the Agents (Graph Nodes)
# ---------------------------------------------------------

async def analyst_agent(state: AgentState):
    print("📊 Analyst Agent: Extracting unique merchants...")
    #standardize_bank_statement("/data/Feb_Statement.xls", "/data/clean_Statement.csv")
    
    df = pd.read_csv("/data/clean_Statement.csv")
    
    # 1. Get the unique merchants and clean them instantly
    df['Clean_Desc'] = df['Description'].astype(str).str.strip().str.lower()
    unique_merchants = df['Clean_Desc'].unique().tolist()
    
    print(f"   -> Found {len(unique_merchants)} unique merchants. Starting Micro-Loop...")
    
    category_dict = {}
    
    # 2. 🚀 THE MICRO-LOOP
    # We ask the base LLM for exactly ONE word per merchant. No JSON needed!
    for merchant in unique_merchants:
        prompt = (
            f"Categorize this bank transaction payee: '{merchant}'.\n"
            "Categories: UPI, Wellness, Groceries, Uncategorized.\n"
            "CRITICAL: Reply with ONLY the exact category word. Do not write any other text, punctuation, or explanations."
        )
        
        try:
            # Note: We use the base 'llm' here, NOT 'structured_llm'
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Clean whatever the LLM spits out (remove spaces, quotes, periods)
            category = response.content.strip().strip('"').strip("'").strip(".")
            category_dict[merchant] = category
            
        except Exception as e:
            category_dict[merchant] = 'Uncategorized'

    print(f"   ✅ Successfully categorized all {len(category_dict)} unique merchants!")

    # 3. Map the categories back to the 107 rows
    print("   -> Stitching categories back into master dataset...")
    df['Category'] = df['Clean_Desc'].map(category_dict).fillna('Uncategorized')
    
    # Clean up the temp column
    df = df.drop(columns=['Clean_Desc'])

    # Count how many failed
    uncategorized_count = len(df[df['Category'] == 'Uncategorized'])
    print(f"   ⚠️ {uncategorized_count} out of {len(df)} transactions are Uncategorized.")

    # 4. Do the flawless math
    total_spent = df['Amount'].sum()
    
    # 5. Build the final JSON payload for your database
    final_payload = {
        "summary": {
            "total_spent": round(total_spent, 2),
            "transaction_count": len(df)
        },
        "transactions": df.to_dict(orient='records')
    }
    
    return {"final_report": json.dumps(final_payload, indent=2)}


def database_agent(state: AgentState):
    print("✅ Database Agent: saving to database...")
    
    # Here you would connect to your database and save the final report
    json_data = state.get("final_report", "{}")
    data_dict = json.loads(json_data)
    
    if data_dict["transactions"]:
        df_to_save = pd.DataFrame(data_dict["transactions"])
        conn = sqlite3.connect('/data/finance.db')
        df_to_save.to_sql('transactions', conn, if_exists='append', index=False)
        conn.close()
        print("💾 Saved to SQLite.")
    return {"final_report": json_data}  # Pass the same report forward if needed
# ---------------------------------------------------------
# 4. Build the LangGraph Workflow
# ---------------------------------------------------------

workflow = StateGraph(AgentState)

# workflow.add_node("reader_agent", reader_agent)
workflow.add_node("analyst_agent", analyst_agent)
workflow.add_node("database_agent", database_agent) # Add our new node

workflow.set_entry_point("analyst_agent") # Start with the Analyst since it handles the reading too
workflow.add_edge("analyst_agent", "database_agent")
workflow.add_edge("database_agent", END)

# Compile graph
app = workflow.compile()


def standardize_bank_statement(input_excel_path: str, output_csv_path: str):
    print(f"🧹 Cleaning data from: {input_excel_path}")
    
    # 1. Read the Excel file
    engine = 'openpyxl' if input_excel_path.endswith('.xlsx') else 'xlrd'
    df = pd.read_excel(input_excel_path, engine=engine, skiprows=20)  # Start from the first row, we'll handle headers flexibly

    # 2. Define aliases to find the right columns
    date_aliases = ['Date', 'Transaction Date', 'Posting Date', 'Post Date']
    desc_aliases = ['Description', 'Payee', 'Transaction Details', 'Name', 'Memo', 'Narration']
    amount_aliases = ['Amount', 'Transaction Amount', 'Value', 'Net Amount', 'Withdrawal Amt.', 'Deposit Amt.']

    actual_columns = df.columns.tolist()
    date_col = next((col for col in actual_columns if col in date_aliases), None)
    desc_col = next((col for col in actual_columns if col in desc_aliases), None)
    amount_col = next((col for col in actual_columns if col in amount_aliases), None)

    # Handle split debit/credit columns if 'Amount' is missing
    if not amount_col:
        debit_col = next((col for col in actual_columns if 'Debit' in str(col)), None)
        credit_col = next((col for col in actual_columns if 'Credit' in str(col)), None)
        if debit_col and credit_col:
            df['Amount'] = df[credit_col].fillna(0) - df[debit_col].fillna(0)
            amount_col = 'Amount'

    if not all([date_col, desc_col, amount_col]):
        raise ValueError(f"Missing required columns. Found: {actual_columns}")

    # Extract our 3 target columns
    clean_df = df[[date_col, desc_col, amount_col]].copy()
    clean_df.columns = ['Date', 'Description', 'Amount']


    # ==========================================
    # 🚀 NEW: THE TRANSACTION FILTERING ENGINE
    # ==========================================
    
    # Filter 1: The Date Test
    # This forces the Date column into actual datetime objects. 
    # If a row has text like "Closing Balance" in the date column, it becomes NaT (Not a Time).
    clean_df['Date'] = pd.to_datetime(clean_df['Date'], errors='coerce', dayfirst=True)  # dayfirst=True is common in many bank statements
    
    # Drop any row that resulted in NaT (This instantly deletes 90% of junk/summary rows!)
    clean_df = clean_df.dropna(subset=['Date'])

    # Filter 2: The Number Test
    # Forces Amount to be a number. Text becomes NaN. Drop the NaNs.
    clean_df['Amount'] = pd.to_numeric(clean_df['Amount'], errors='coerce')
    clean_df = clean_df.dropna(subset=['Amount'])

    # Filter 3: The Keyword Test
    # Sometimes banks put a valid date next to the "Opening Balance". We explicitly block these.
    # We use .fillna('') to prevent errors if the description is blank.
    ignore_keywords = ['opening balance', 'closing balance', 'ending balance', 'statement balance']
    
    # Convert descriptions to lowercase and check if they match our ignore list
    clean_df = clean_df[~clean_df['Description'].fillna('').str.lower().str.strip().isin(ignore_keywords)]
    
    # Remove rows where the amount is exactly 0.00
    clean_df = clean_df[clean_df['Amount'] != 0.0]

    # ==========================================
    clean_df = clean_df.iloc[:-1]  # Drop the last row which often contains "Closing Balance" or similar summary info
    # Save the pristine, filtered data to CSV
    clean_df.to_csv(output_csv_path, index=False)
    print(f"✅ Success! Filtered to {len(clean_df)} valid transactions and saved.")

def process_all_statements(data_dir: str, final_csv_path: str):
    print("📂 Scanning for new bank statements...")
    
    # Find all Excel files in the root of the data folder
    search_pattern = os.path.join(data_dir, "*.xls*")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print("   -> No new statements found.")
        return False # Return False so we know to skip the LLM agent entirely

    master_df_list = []
    
    # Ensure processed directory exists
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Loop through and clean every file
    for file_path in file_list:
        print(f"   -> Cleaning: {os.path.basename(file_path)}")
        try:
            standardize_bank_statement(file_path, "temp_clean.csv")
            clean_df = pd.read_csv("temp_clean.csv")
            
            master_df_list.append(clean_df)
            
            # 2. MOVE the file to the processed folder so we don't read it tomorrow
            new_location = os.path.join(processed_dir, os.path.basename(file_path))
            shutil.move(file_path, new_location)
            
        except Exception as e:
            print(f"   ❌ Failed to process {os.path.basename(file_path)}: {str(e)}")

    # 3. Smash them all together and save
    if master_df_list:
        final_df = pd.concat(master_df_list, ignore_index=True)
        final_df.to_csv(final_csv_path, index=False)
        print(f"✅ Success! Consolidated {len(master_df_list)} files into {len(final_df)} total transactions.")
        return True
    
    return False

async def run_pipeline():
    print("🚀 Starting Local CFO pipeline...")
    start_time = time.time() 

    # 1. Run the Bulk Processor First
    has_new_data = process_all_statements("/data", "/data/clean_statement.csv")
    
    # 2. Only run the LLM if we actually found new statements
    if has_new_data:
        # Run your graph
        final_state = await app.ainvoke({"current_index": 0})
        
        end_time = time.time()
        print(final_state["final_report"])
        print(f"\n⏱️ DONE! Processed new files in {int(end_time - start_time)} seconds.")
    else:
        print("💤 No new data to process. Going back to sleep.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())