#!/bin/bash
# Simple setup script to install dependencies for running tests
set -e
if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found in current directory" >&2
  exit 1
fi
python -m pip install --upgrade pip
pip install -r requirements.txt
