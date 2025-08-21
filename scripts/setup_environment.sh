#!/bin/bash

# NHL Draft Prediction Project - Environment Setup Script
# This script sets up the development environment for the project

echo "Setting up NHL Draft Prediction Project environment..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is installed"
else
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed
mkdir -p models/saved
mkdir -p results/figures results/tables
mkdir -p tests
mkdir -p docs

echo "✅ Project directories created"

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "ℹ️  pre-commit not installed. Install with: pip install pre-commit"
fi

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('✅ NLTK data downloaded')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the EDA notebook: jupyter notebook notebooks/01_eda.ipynb"
echo "3. Check the README.md for more information"
echo ""
echo "Happy coding! 🏒" 