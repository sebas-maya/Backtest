#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# TO EXECUTE IN TERMINAL
# chmod +x run_app.command

# ACTIVATE VIRTUAL ENV
source "$SCRIPT_DIR/.venv/bin/activate"

# RUN STREAMLIT FILE
python3 -m streamlit run "$SCRIPT_DIR/app/Home.py"
