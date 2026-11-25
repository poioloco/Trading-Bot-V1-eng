#!/bin/bash
# === Tactical Trading Bot - launcher (Linux) ===
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "First run: setting up the environment, one moment..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
else
  source .venv/bin/activate
fi
python menu.py
