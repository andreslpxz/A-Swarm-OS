#!/bin/bash
set -e

# Format and Lint could go here
echo "Running Pre-commit checks..."

# Check Python Syntax
python -m py_compile swarm_os/**/*.py

echo "Pre-commit checks passed!"
