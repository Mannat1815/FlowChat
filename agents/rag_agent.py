# rag_agent.py (Updated with anti-repetition prompt and better model)
# agents/rag_agent.py (Updated with retries and increased timeout to fix ReadTimeout)
from agents.indexing_agent import query_similar
import requests
import os
from dotenv import load_dotenv
from retrying import retry  # Add this for retries

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# FIX: Changing model to match app.py's successful configuration
DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free") 

def textwrap_shorten(s, length):
    if not s:
        return ""
    return s if len(s) <= length else s[:length-3] + "..."

def build_prompt(query, contexts):
    """
    Builds the RAG prompt. If context is retrieved, use it for answering. 
    If context is empty, use a general knowledge prompt about ARGO data.
    """
    ctx_text = "\n\n---\n".join(
        [f"Source: {c['source']}\n{textwrap_shorten(c['text'], 500)}" for c in contexts]
    )
    
    if not contexts:
        # Fallback to general knowledge prompt
        general_context = """
        You are an oceanographic data expert. If no specific database context is retrieved, provide a general, helpful, and concise answer about ARGO data, oceanographic parameters, and the capabilities of this FloatChat system.
        The FloatChat system uses a PostgreSQL database to store ARGO profile (1D: LATITUDE, LONGITUDE, JULD, source_file) and measurement (2D: pres, temp, psal) data.
        If the user asks for a plot or visualization, instruct them to use the **Visualization tab (Tab 4)**.
        """
        prompt = f"""
{general_context.strip()}

USER QUESTION: {query}

Answer concisely in a helpful, knowledgeable, and non-repetitive way.
"""
    else:
        # Use retrieved context for specific factual answering
        prompt = f"""
You are an assistant answering oceanographic data questions.

RETRIEVED CONTEXTS:
{ctx_text}

USER QUESTION:
{query}

Answer concisely using the contexts. Do not repeat any content. Provide a single concise answer. Do not include SQL queries.
"""
    
    # Post-process: Attempt to remove redundant content that LLM sometimes appends
    # This is handled in the final output in app.py now, but keeping this clean helps.
    return prompt

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=3)
def call_llm(prompt, model=None):
    model = model or DEFAULT_MODEL
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    # Increased timeout to 120 seconds
    r = requests.post(url, headers=headers, json=payload, timeout=120) 
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    
    # Clean up common LLM repetition patterns just in case
    content = content.replace("Answer concisely using the contexts.", "").strip()
    return content

def answer_with_rag(user_query, top_k=5):
    contexts = query_similar(user_query, top_k=top_k)
    prompt = build_prompt(user_query, contexts)
    try:
        answer = call_llm(prompt)  # Now with retries
    except Exception as e:
        answer = f"Error calling LLM: {e}"
    return {"answer": answer, "contexts": contexts}
