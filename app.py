# app.py (Updated with SQL stripping, updated prompt, and column case fixes)
# app.py (Modified with all fixes and restored visualization)
import os
import streamlit as st
import pandas as pd
import tempfile
import gc
import time
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import re  # Added for SQL stripping

# LangChain and LangGraph imports for multi-agent workflow
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
import operator

# Agents (import existing ones)
from agents.ingest_agent import ingest_file
# The original translate_to_sql function from query_translator_agent.py is not used in the core flow
from agents.rag_agent import answer_with_rag
from agents.visualization_agent import visualization_agent, get_plotly_figure_html # Import needed functions
from agents.indexing_agent import index_records
from agents.blockchain_agent import log_ingestion_event

load_dotenv()

# Postgres config (same fallback)
PG_HOST = os.getenv("POSTGRES_HOST") or os.getenv("MYSQL_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT") or os.getenv("MYSQL_PORT", "5432")
PG_USER = os.getenv("POSTGRES_USER") or os.getenv("MYSQL_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD") or os.getenv("MYSQL_PASSWORD", "")
PG_DB = os.getenv("POSTGRES_DB") or os.getenv("MYSQL_DATABASE") or "argo_db"

PG_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(PG_URL, pool_pre_ping=True)

# LangChain SQLDatabase setup
db = SQLDatabase(engine=engine)

# LLM setup (using ChatOpenAI with OpenRouter base URL)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# FIX: Model causing 404 error is replaced with a stable free model.
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

llm = ChatOpenAI(
    model=LLM_MODEL, 
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Improved prompt for SQL translation (FIXED for JOINING LOGIC and template variables)
IMPROVED_SQL_PROMPT = PromptTemplate.from_template("""
You are a PostgreSQL expert. Given the database schema below and the user question, generate a single, valid SQL query.

Database Schema:
{table_info}
argo_profiles (p): id, source_file, ingest_ts, LATITUDE, LONGITUDE, JULD (uppercase).
argo_measurements (m): id, source_file, profile_id (FK to argo_profiles), level_index, pres, temp, psal (lowercase).
argo_nd_values (n): id, source_file, var_name, value, dim0, dim1... (for general N-D data)

Guidelines:
- **STRICT**: Only output one complete SQL query. Do NOT repeat the query or include explanations.
- **JOINS**: If the query requires both profile information (LATITUDE, LONGITUDE, JULD) and measurement data (pres, temp, psal), you **MUST** join the tables on `p.id = m.profile_id`. Use aliases (p, m, n) for brevity.
- For aggregations (AVG, MIN, MAX), use appropriate functions.
- Do not use LIMIT for aggregations without GROUP BY.
- Always include LIMIT {top_k} for non-aggregation SELECT queries unless explicitly requested otherwise.
- **FILE FILTER**: When filtering by float ID (e.g., '5904550'), the value in the 'source_file' column usually includes the '.nc' extension (e.g., '5904550.nc'). Use this format in your WHERE clause.
- **COLUMN CASE**: Use uppercase for argo_profiles (e.g., LATITUDE), and lowercase for argo_measurements (e.g., pres, temp, psal).

User Question: {input}
SQL Query:
""")

# LangChain chain for SQL query generation
# NOTE: The missing 'table_info' variable is implicitly handled by create_sql_query_chain
# but including it explicitly in the template above (and here for safety) ensures compatibility.
sql_chain = create_sql_query_chain(llm, db, prompt=IMPROVED_SQL_PROMPT)

# Improved RAG prompt for better output (NOTE: This prompt is technically redundant 
# since the logic is now handled in rag_agent.py's build_prompt function, 
# but we keep it structured here for LangChain compatibility.)
IMPROVED_RAG_PROMPT = """
You are an expert oceanographic data analyst. Use the following retrieved contexts to answer the user's question accurately and concisely.

Retrieved Contexts:
{contexts}

User Question: {query}

Provide a clear, factual answer in a single paragraph. If the question requires SQL, suggest a query. Include key metadata like sources and scores if relevant.
"""

# Define state for LangGraph multi-agent workflow
class AgentState(TypedDict):
    query: str
    sql: Annotated[str, operator.add]
    rag_answer: Annotated[str, operator.add]
    results: pd.DataFrame
    visualization: str  # RESTORED visualization placeholder

# Define agents as nodes in LangGraph
def rag_agent(state: AgentState) -> AgentState:
    rag_out = answer_with_rag(state["query"])
    state["rag_answer"] = rag_out["answer"] if rag_out else "No RAG response."
    return state

def sql_translator_agent(state: AgentState) -> AgentState:
    # FIX: Revert to 'question' key. Although the prompt uses '{input}', 
    # the internal components of create_sql_query_chain expect 'question' as input.
    sql = sql_chain.invoke({"question": state["query"], "top_k": 100})
    # New: Strip markdown code blocks if present
    sql = sql.strip()
    if sql.startswith('```sql'):
        sql = sql.split('```sql', 1)[1].split('```', 1)[0].strip()
    elif sql.startswith('```'):
        sql = sql.split('```', 1)[1].split('```', 1)[0].strip()
    state["sql"] = sql
    return state

def executor_agent(state: AgentState) -> AgentState:
    try:
        # Split into commands and execute only the first one to handle LLM duplication errors
        sql_commands = [cmd.strip() for cmd in state["sql"].split(';') if cmd.strip()]
        
        if not sql_commands:
            state["results"] = pd.DataFrame()
            state["rag_answer"] += "\n\n**WARNING**: LLM returned no executable SQL."
            return state

        with engine.connect() as conn:
            # Execute only the FIRST SQL statement
            res = conn.execute(text(sql_commands[0]))
            # Use fetchall() safely
            rows = res.fetchall()
            
            # Only try to convert to dict/DataFrame if rows exist.
            if rows:
                 # Check if the result set has column names (typical of SELECT)
                try:
                    state["results"] = pd.DataFrame([dict(r) for r in rows])
                except Exception:
                    # Fallback for simple results (e.g., COUNT(*)) or non-dict results
                    state["results"] = pd.DataFrame(rows, columns=res.keys())
            else:
                 state["results"] = pd.DataFrame() # Empty DataFrame for non-SELECT or no results
            
            if len(sql_commands) > 1:
                 state["rag_answer"] += f"\n\n**WARNING**: LLM returned {len(sql_commands)} queries. Only the first one was executed."
    except Exception as e:
        # Format the SQL error directly into the results message
        state["results"] = pd.DataFrame()
        state["rag_answer"] += f"\n\n**ERROR**: Failed to execute SQL query '{state['sql'][:50]}...'. Details: {e}"
    return state

def visualizer_agent(state: AgentState) -> AgentState:
    # Visualization is handled interactively in Streamlit tab4, 
    # but this node remains for workflow continuity.
    state["visualization"] = "Visualization flagged for manual generation."
    return state

# Build LangGraph workflow
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("rag", rag_agent)
workflow.add_node("sql_translator", sql_translator_agent)
workflow.add_node("executor", executor_agent)
workflow.add_node("visualizer", visualizer_agent) 

# Edges: Simple sequential
workflow.add_edge(START, "rag")
workflow.add_edge("rag", "sql_translator")
workflow.add_edge("sql_translator", "executor")
workflow.add_edge("executor", "visualizer") 
workflow.add_edge("visualizer", END)

# Compile graph
app_graph = workflow.compile()

# Database and Session Cleanup Function
def clear_session_and_db():
    try:
        with engine.begin() as conn:
            # Delete data from all tables
            conn.execute(text("DELETE FROM argo_measurements;"))
            conn.execute(text("DELETE FROM argo_profiles;"))
            # Assuming ingest_agent also uses 'argo_nd_values' if AUTO_EXPLODE_ND is true
            conn.execute(text("DELETE FROM argo_nd_values;"))
        st.success("Successfully cleared all data from argo_profiles, argo_measurements, and argo_nd_values tables.")
    except Exception as e:
        st.error(f"Error clearing database: {e}")

    # Clear session state variables
    st.session_state.ingested_files = []
    st.session_state.chat_history = []
    
    # Reload the page to reset the Streamlit session
    st.rerun()

# Streamlit app setup
st.set_page_config(page_title="FloatChat (Postgres + Multi-Agent)", layout="wide")
st.title("🌊 FloatChat – Multi-Agent ARGO Explorer (Postgres)")

# Initialize session state for ingested files
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2, tab3, tab4 = st.tabs(["Ingest Data", "Data Summary", "Chatbot", "Visualization"])

with tab1:
    st.header("Ingest Data")
    st.write("Steps for Ingestion:")
    st.write("1. Upload one or more NetCDF files (.nc).")
    st.write("2. Files will be parsed and ingested into Postgres.")
    st.write("3. Optional: Automatically explode N-D arrays and index chunks for RAG.")
    st.write("4. Use the 'Clear Data' button to reset the database and session.")

    uploaded_files = st.file_uploader("Upload NetCDF files", type=["nc"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.ingested_files:
                st.warning(f"File '{uploaded_file.name}' already ingested. Skipping.")
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                success, msg = ingest_file(tmp_path)
                os.unlink(tmp_path)
                if success:
                    st.success(f"Successfully ingested {uploaded_file.name}: {msg}")
                    st.session_state.ingested_files.append(uploaded_file.name)
                    gc.collect()
                else:
                    st.error(f"Failed to ingest {uploaded_file.name}: {msg}")

    if st.button("Clear Data"):
        with st.spinner("Clearing database and session..."):
            clear_session_and_db()

with tab2:
    st.header("Data Summary")
    st.write("Steps for Summary:")
    st.write("1. Select a specific file or 'Overall'.")
    st.write("2. Generate a summary of the data in the database (counts, columns, stats).")

    ingested_files = ["Overall"] + st.session_state.ingested_files
    selected_file = st.selectbox("Select File for Summary:", ingested_files)
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            # Setup WHERE clauses and params
            if selected_file == "Overall":
                where_clause = ""
                params = {}
                measurements_where = ""
            else:
                where_clause = "WHERE source_file = :sf"
                params = {"sf": selected_file}
                measurements_where = "WHERE source_file = :sf"

            with engine.connect() as conn:
                # 1. Profile (1D) Data Summary (argo_profiles)
                count_profiles = conn.execute(text(f"SELECT COUNT(*) FROM argo_profiles {where_clause}"), params).scalar()
                columns = [row[0] for row in conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'argo_profiles'")).fetchall()]

                st.write(f"Summary for {selected_file}:")
                st.write(f"- Total profiles in argo_profiles: {count_profiles}")
                st.write(f"- Columns in argo_profiles: {', '.join(columns)}")

                # 2. Measurement (2D) Data Summary (argo_measurements)
                try:
                    stats_query = text(f"SELECT MIN(temp) as min_temp, MAX(temp) as max_temp, AVG(temp) as avg_temp, COUNT(*) as row_count FROM argo_measurements {measurements_where}")
                    stats = conn.execute(stats_query, params).fetchone()

                    if stats and stats[0] is not None: # Check first element of tuple/row
                         st.write(f"- Total rows in argo_measurements: {stats.row_count}")
                         st.write(f"- **Temperature (argo_measurements) stats**: Min {stats.min_temp:.3f}, Max {stats.max_temp:.3f}, Avg {stats.avg_temp:.3f}")
                    else:
                         st.warning("No temperature data found in argo_measurements for selected scope.")

                except Exception as e:
                    st.error(f"Error querying argo_measurements: {e}")

with tab3:
    st.header("Chatbot (Paragraph Answers with Process in Sidebar)")
    st.write("Steps for Chatbot:")
    st.write("1. Enter a natural language query (e.g., 'What is the average salinity?').")
    st.write("2. Query is routed through multi-agent graph: RAG -> SQL Translation -> Execution.")
    st.write("3. Main screen shows clean paragraph-style answer.")
    st.write("4. Right sidebar shows process details (RAG, SQL, Tabular Results).")
    st.write("5. Chat history is maintained.")

    # Display chat history (only user queries and clean paragraph answers in main screen)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about the data...")
    if user_query:
        with st.spinner("Processing query..."):
            # Run LangGraph workflow
            inputs = {"query": user_query}
            output = app_graph.invoke(inputs)
            
            # --- Chatbot Output Generation ---
            
            # Clean paragraph-style answer for main screen (primarily RAG answer, with brief results summary if available)
            main_answer = output['rag_answer']
            if not output['results'].empty:
                # Append a brief summary of results to the paragraph
                
                # Filter out columns with all NaNs before describing, ensuring only meaningful stats are used
                numeric_df = output['results'].select_dtypes(include=['number'])
                # Drop columns where all values are NaN
                clean_df = numeric_df.dropna(axis=1, how='all')

                if not clean_df.empty:
                    # Generate summary for cleaned dataframe
                    summary_data = clean_df.describe().iloc[[1, 2, 3, 7]]
                    
                    # Use intermediate variable for replace to avoid backslash syntax error
                    summary_string = summary_data.to_string().replace('\n', ' | ')
                    results_summary = f" The executed query returned {len(output['results'])} data rows. Results for key metrics are: {summary_string}."
                    main_answer += results_summary
                
            
            # If the main answer is just the standard "No RAG response" (which it shouldn't be with the fix), skip it.
            if main_answer.strip() == "No RAG response.":
                main_answer = "I could not find a specific answer in the available data contexts."
            
            # Display user query and assistant response
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": main_answer})
            
            # Manually re-display history to include the new messages (Streamlit redraw logic)
            # Find the index of the last displayed message and append the new ones
            for message in st.session_state.chat_history[-2:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Display process details in the sidebar (clean and separate)
            with st.sidebar:
                st.header("Process Details")
                st.markdown("---")

                # RAG Answer (raw output from RAG agent)
                st.subheader("RAG Agent Response")
                st.markdown(output['rag_answer'])

                # SQL Query
                st.subheader("SQL Query")
                st.code(output['sql'].strip(), language="sql")

                # Tabular Results
                st.subheader("Tabular Results")
                if not output['results'].empty:
                    st.dataframe(output['results'])
                else:
                    st.warning("No data found or query failed.")

                st.markdown("---")

with tab4:
    st.header("Visualization")
    st.write("Steps for Visualization:")
    st.write("1. Enter a query prompt (natural language) to retrieve data for plotting.")
    st.write("2. Select a plot style (Line Profile, Scatter Map, Histogram, or Box Plot).")
    st.write("3. Generate plots and download HTML output.")

    viz_prompt = st.text_input("Enter natural language prompt for visualization query (e.g., 'Select temperature, salinity and depth for all measurements')")
    
    plot_style = st.selectbox(
        "Select Plot Style:",
        options=[
            "line_depth", 
            "scatter_profile", 
            "histogram", 
            "box_plot",
            "scatter_map"
        ],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if st.button("Generate Visualization"):
        if viz_prompt:
            with st.spinner("Translating query..."):
                # 1. Translate NL prompt to SQL using the LLM chain
                try:
                    # Reuse the configured sql_chain
                    # NOTE: Here we pass 'question' and 'top_k' keys.
                    final_sql = sql_chain.invoke({"question": viz_prompt, "top_k": 500})
                    
                    # Strip markdown block structure (```sql ... ```)
                    final_sql = final_sql.strip()
                    if final_sql.startswith('```sql'):
                        final_sql = final_sql.split('```sql', 1)[1].split('```', 1)[0].strip()
                    elif final_sql.startswith('```'):
                        final_sql = final_sql.split('```', 1)[1].split('```', 1)[0].strip()
                    
                    # Remove trailing semicolon for manipulation if present
                    if final_sql.endswith(';'):
                        final_sql = final_sql[:-1].strip()

                    # --- FIX: Inject required columns if needed for profile/map plots ---
                    
                    # 1. Determine columns required for the selected plot type
                    required_cols = set()
                    if plot_style in ["line_depth", "scatter_profile"]:
                        required_cols.add("pres") # Renamed to depth later
                        required_cols.add("profile_id") # Needed for grouping profiles
                    if plot_style == "scatter_map":
                        required_cols.add("LATITUDE")
                        required_cols.add("LONGITUDE")
                        required_cols.add("profile_id")
                    
                    # 2. Extract the SELECT clause columns from the LLM-generated query
                    # This is brittle but necessary for modification. Assuming standard 'SELECT <cols> FROM ...' structure.
                    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', final_sql, re.IGNORECASE)
                    
                    if select_match:
                        current_cols_str = select_match.group(1).strip()
                        current_cols = {col.split('.')[-1].strip().upper().replace('"', '') for col in current_cols_str.split(',')}

                        # 3. Add missing required columns
                        cols_to_add = []
                        for req_col in required_cols:
                            # Use uppercase for check since LLM prompt uses uppercase for profiles (LATITUDE)
                            # and lowercase for measurements (pres)
                            check_name = req_col.upper()
                            
                            if check_name not in current_cols and req_col.lower() not in current_cols:
                                # Determine correct alias and case for injection
                                if req_col in ["pres", "profile_id"]:
                                    cols_to_add.append(f"m.{req_col}") # Measurement table
                                elif req_col in ["LATITUDE", "LONGITUDE"]:
                                    cols_to_add.append(f"p.{req_col}") # Profile table
                                else:
                                    # Fallback, just add the column name without alias
                                    cols_to_add.append(req_col)
                        
                        if cols_to_add:
                            # Reconstruct the SELECT clause
                            new_cols_str = current_cols_str + ", " + ", ".join(cols_to_add)
                            # Re-inject the modified SELECT clause back into the SQL
                            final_sql = final_sql.replace(current_cols_str, new_cols_str, 1).strip()
                            
                            # Ensure JOIN is present if we are adding profile or measurement columns
                            if any(col.startswith(('m.', 'p.')) for col in cols_to_add) or "profile_id" in required_cols:
                                if "JOIN" not in final_sql.upper():
                                    # Very basic attempt to insert JOIN: Find the first FROM, and append a simple join
                                    from_match = re.search(r'FROM\s+.*?(\s+WHERE|\s+GROUP BY|\s+ORDER BY|\s+LIMIT|$)', final_sql, re.IGNORECASE)
                                    if from_match:
                                        insert_point = from_match.start(1) if from_match.group(1) else len(final_sql)
                                        # Assuming we start with the measurements table (m) and need to join profiles (p)
                                        # This injection assumes the query starts from one table and we are adding the other.
                                        
                                        # Since LLM is prompted to use aliases (p) and (m), we check which is present
                                        if 'm.' in final_sql.lower() and 'p.' not in final_sql.lower():
                                            # We need to join p
                                            join_clause = " JOIN argo_profiles p ON p.id = m.profile_id "
                                            final_sql = final_sql[:insert_point] + join_clause + final_sql[insert_point:].strip()
                                        elif 'p.' in final_sql.lower() and 'm.' not in final_sql.lower():
                                            # We need to join m
                                            join_clause = " JOIN argo_measurements m ON p.id = m.profile_id "
                                            final_sql = final_sql[:insert_point] + join_clause + final_sql[insert_point:].strip()
                                        elif 'argo_profiles' in final_sql.lower() and 'argo_measurements' not in final_sql.lower():
                                            # No aliases, but primary table is profiles. Join measurements. (Less reliable)
                                             join_clause = " JOIN argo_measurements ON argo_profiles.id = argo_measurements.profile_id "
                                             final_sql = final_sql[:insert_point] + join_clause + final_sql[insert_point:].strip()

                            st.warning(f"Automatically injected columns: {', '.join(cols_to_add)}")

                    # --- End FIX: Inject required columns ---
                    
                    st.code(final_sql.strip(), language="sql") 
                except Exception as e:
                    st.error(f"Error in query translation: {e}")
                    final_sql = None
            
            if final_sql:
                # 2. Execute the generated SQL
                try:
                    with engine.connect() as conn:
                        # Execute only the first query if LLM repeats
                        sql_commands = [cmd.strip() for cmd in final_sql.split(';') if cmd.strip()]
                        
                        res = conn.execute(text(sql_commands[0]))
                        rows = res.fetchall() # fetchall() returns a list of Row objects or tuples
                        
                        if rows:
                            # Use keys from result set to construct DataFrame
                            column_names = res.keys()
                            df = pd.DataFrame(rows, columns=column_names)
                            
                            # Rename columns for visualization consistency (temp->temperature, pres->depth, psal->salinity)
                            # Also rename LATITUDE, LONGITUDE, JULD to lowercase for visualization agent
                            rename_map = {
                                "pres": "depth", 
                                "temp": "temperature", 
                                "psal": "salinity", 
                                "juld": "time",
                                "LATITUDE": "latitude",
                                "LONGITUDE": "longitude",
                                "JULD": "time" # JULD is already mapped, ensure consistent column key
                            }
                            # Only apply renames for columns that actually exist in the DataFrame
                            rename_map_filtered = {k: v for k, v in rename_map.items() if k in df.columns}
                            df = df.rename(columns=rename_map_filtered) 


                            st.subheader("Query Results (First 5 Rows)")
                            st.dataframe(df.head())
                            st.subheader("Visualizations")
                            
                            # Call the agent, which handles plotting multiple figures based on data
                            figures = visualization_agent(df, plot_style)

                            # Add Download options
                            if figures:
                                st.markdown("---")
                                st.subheader("Download Options")
                                cols = st.columns(max(1, len(figures))) # Ensure at least one column
                                for i, (name, fig) in enumerate(figures.items()):
                                    with cols[i % len(cols)]:
                                        fig_html_data = get_plotly_figure_html(fig)
                                        st.download_button(
                                            label=f"Download {name}.html",
                                            data=fig_html_data,
                                            file_name=f"{name}_{plot_style}.html",
                                            mime="text/html"
                                        )
                            else:
                                st.warning("Visualization agent returned no figures. Check if required columns (e.g., 'depth', 'temperature') are present.")
                        else:
                            st.warning("No data returned by the executed query.")
                except Exception as e:
                    st.error(f"Error executing SQL query for visualization: {e}")

st.markdown("---")
st.write("This application uses LangChain for SQL chains and improved prompts, LangGraph for multi-agent orchestration (RAG -> SQL -> Execute -> Visualize), and tabs for UI separation.")
