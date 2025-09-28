# agents/blockchain_agent.py
import os
import hashlib
import time
import json

def log_ingestion_event(file_path):
    """
    Simulate logging an ingestion event to blockchain by writing a JSON
    event to a local 'ledger' file (or printing). This shows provenance.
    """
    with open(file_path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    event = {
        "timestamp": int(time.time()),
        "file": os.path.basename(file_path),
        "sha256": h
    }
    ledger_path = os.getenv("LEDGER_PATH", "blockchain_ledger.jsonl")
    with open(ledger_path, "a") as lf:
        lf.write(json.dumps(event) + "\n")
    print("Logged ingestion event to ledger:", event)
