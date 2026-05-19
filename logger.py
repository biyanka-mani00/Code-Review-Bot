import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

LOG_DIR = Path("logs")

def log_session(
    filename: str, 
    model: str, 
    input_tokens: int, 
    output_tokens: int, 
    latency_ms: int, 
    response: str
) -> None:
    """
    Logs a code review session to a daily JSON file.
    """
    LOG_DIR.mkdir(exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"{today}.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "response": response
    }
    
    logs = []
    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
        except (json.JSONDecodeError, IOError):
            logs = []
            
    logs.append(log_entry)
    
    try:
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    except IOError as e:
        print(f"Error writing to log file: {e}")
