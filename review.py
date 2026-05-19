#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from client import stream_review
from logger import log_session

# --- Configuration ---
SYSTEM_INSTRUCTION = (
    "You are a senior software engineer and code reviewer. "
    "Analyze the provided code for bugs, security vulnerabilities, "
    "performance issues, and adherence to best practices. "
    "Provide clear, actionable feedback."
)

def run_pipeline(file_path: Path):
    """
    Executes the review pipeline as a chain of steps.
    """
    # STEP 1: Read the file
    print(f"\n[1/4] Reading file: {file_path.name}...")
    if not file_path.is_file():
        print(f"Error: File '{file_path}' not found.")
        return
    content = file_path.read_text(encoding='utf-8')

    # STEP 2: Call client and stream live response
    print(f"[2/4] Requesting review from Gemini...\n")
    print("="*60)
    
    metadata = {}
    try:
        for chunk in stream_review(
            file_name=file_path.name,
            content=content,
            system_instruction=SYSTEM_INSTRUCTION,
            metadata=metadata
        ):
            print(chunk, end="", flush=True)
        
        print("\n" + "="*60)

        # STEP 3: Print token counts and latency
        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        latency = metadata.get("latency_ms", 0)

        print(f"\n[3/4] Review Complete.")
        print(f"      - Input Tokens:  {input_tokens}")
        print(f"      - Output Tokens: {output_tokens}")
        print(f"      - Latency:       {latency}ms")

        # STEP 4: Call logger
        print(f"[4/4] Saving session to logs...")
        log_session(
            filename=file_path.name,
            model=metadata.get("model", "unknown"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            response=metadata.get("response", "")
        )
        print("Done!")

    except Exception as e:
        print(f"\nPipeline failed at Step 2: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='AI Code Review Bot')
    parser.add_argument('file_path', type=Path, help='Path to the file to review')
    args = parser.parse_args()

    run_pipeline(args.file_path)

if __name__ == "__main__":
    main()
