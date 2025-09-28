import os
import json
import requests
from dotenv import load_dotenv

# -------------------- SETUP --------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")

# -------------------- TRANSLATE TO SQL --------------------
def translate_to_sql(user_query, schema_hint=None, model_name=None):
    """
    Translates a natural language user query into a PostgreSQL SQL query 
    using an external LLM (OpenRouter).
    """
    model = model_name or DEFAULT_MODEL
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in env")

    if not schema_hint:
        # Provide schema hint for the LLM model
        schema_hint = """
        You are a PostgreSQL expert. The database contains oceanographic data across two main tables:
        argo_profiles (1D profile metadata): Columns include id, source_file, ingest_ts, latitude, longitude, and 1D variables like juld (time).
        argo_measurements (2D depth level data): Columns include id, source_file, profile_id (FK to argo_profiles), level_index, pres (pressure), temp (temperature), psal (salinity).
        
        IMPORTANT GUIDELINES:
        - Use PostgreSQL syntax.
        - For year filtering on argo_profiles.juld use: EXTRACT(YEAR FROM juld) = <year>
        - Use LIMIT to restrict large result sets, e.g., LIMIT 100.
        - Provide only a JSON object: {"query": "<SQL>"}.
        """

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": schema_hint},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    out = resp.json()
    content = out["choices"][0]["message"]["content"]

    # Attempt to parse the SQL query from the LLM response
    try:
        j = json.loads(content)
        return j.get("query")
    except Exception:
        # Fallback for models that output code blocks
        if "```sql" in content:
            return content.split("```sql", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            return content.split("```", 1)[1].split("```", 1)[0].strip()
        return content.strip()
