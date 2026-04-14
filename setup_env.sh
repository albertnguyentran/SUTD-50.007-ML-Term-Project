#!/usr/bin/env bash
set -e

VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

echo "Activating and installing dependencies..."
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name ml-term-project --display-name "Python (ml-term-project)"

echo ""
echo "Done. Select the 'Python (ml-term-project)' kernel in Jupyter."
