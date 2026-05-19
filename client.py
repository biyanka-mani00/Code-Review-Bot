import os
import json
import time
import httpx
from typing import Generator, Dict, Any, Optional

# --- Constants ---
MODEL_NAME = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

def get_api_key() -> str:
    """Retrieves the API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    return api_key

def create_payload(file_name: str, content: str, system_instruction: str) -> Dict[str, Any]:
    """
    Constructs the JSON payload for the Gemini API.
    """
    return {
        "contents": [{
            "parts": [{"text": f"Please review this file: {file_name}\n\nCode content:\n\n{content}"}]
        }],
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        }
    }

def get_stream_url(api_key: str) -> str:
    """
    Constructs the streaming endpoint URL.
    """
    return f"{BASE_URL}/{MODEL_NAME}:streamGenerateContent?alt=sse&key={api_key}"

def parse_sse_line(line: str) -> Generator[Dict[str, Any], None, None]:
    """
    Parses a single SSE data line and yields either text or usage metadata.
    """
    if not line.startswith("data: "):
        return

    data_str = line[6:].strip()
    if not data_str:
        return
    
    try:
        data = json.loads(data_str)
        
        # Check for usage metadata (usually in the last chunk)
        if "usageMetadata" in data:
            yield {"type": "usage", "data": data["usageMetadata"]}

        # Check for text content
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                part = candidate["content"]["parts"][0]
                if "text" in part:
                    yield {"type": "text", "data": part["text"]}
                    
    except json.JSONDecodeError:
        pass

def stream_review(file_name: str, content: str, system_instruction: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
    """
    Main entry point for streaming a code review.
    Handles retries and orchestrates the API request.
    If metadata dict is provided, it will be populated with session info.
    """
    api_key = get_api_key()
    url = get_stream_url(api_key)
    payload = create_payload(file_name, content, system_instruction)

    max_retries = 5
    base_delay = 1.0
    
    start_time = time.time()
    full_response = []

    for attempt in range(max_retries):
        try:
            with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
                if response.status_code == 429:
                    response.read()
                    raise httpx.HTTPStatusError("Rate limit exceeded (429). Please wait a moment before trying again.", request=response.request, response=response)
                
                if 500 <= response.status_code < 600:
                    response.read()
                    raise httpx.HTTPStatusError(f"Server Error ({response.status_code}).", request=response.request, response=response)
                
                response.raise_for_status()

                for line in response.iter_lines():
                    for item in parse_sse_line(line):
                        if item["type"] == "text":
                            text = item["data"]
                            full_response.append(text)
                            yield text
                        elif item["type"] == "usage" and metadata is not None:
                            metadata["input_tokens"] = item["data"].get("promptTokenCount", 0)
                            metadata["output_tokens"] = item["data"].get("candidatesTokenCount", 0)
                
                if metadata is not None:
                    metadata["latency_ms"] = int((time.time() - start_time) * 1000)
                    metadata["response"] = "".join(full_response)
                    metadata["model"] = MODEL_NAME
                
                return

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            print(f"\n[Client] API error ({e}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
