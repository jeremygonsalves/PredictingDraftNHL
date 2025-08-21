#!/usr/bin/env python3
"""
Script to set up the data directory structure and provide instructions.
"""

import os
from pathlib import Path

def setup_data_structure():
    """Set up the data directory structure."""
    
    # Create directories
    directories = [
        "data/raw",
        "data/processed",
        "models/saved",
        "results/figures",
        "results/tables"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create placeholder files
    placeholders = [
        ("data/raw/README.md", "Place your raw data files here (e.g., prospect-data.csv)"),
        ("data/processed/README.md", "Processed data files will be saved here"),
        ("models/saved/README.md", "Trained models will be saved here"),
        ("results/figures/README.md", "Generated plots and visualizations will be saved here"),
        ("results/tables/README.md", "Generated tables and results will be saved here")
    ]
    
    for file_path, content in placeholders:
        with open(file_path, 'w') as f:
            f.write(f"# {Path(file_path).parent.name.title()} Directory\n\n{content}\n")
        print(f"✅ Created placeholder: {file_path}")

def print_instructions():
    """Print setup instructions."""
    
    print("\n" + "="*60)
    print("📋 DATA SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. 📁 Data Files Required:")
    print("   - prospect-data.csv (raw NHL draft data)")
    print("   - reports_with_bert_embeddings.csv (processed BERT embeddings)")
    print("   - reports_with_embeddings.csv (processed Word2Vec embeddings)")
    
    print("\n2. 📂 Place your data files in:")
    print("   - Raw data: data/raw/")
    print("   - Processed data: data/processed/")
    
    print("\n3. 🔧 If you don't have the data files:")
    print("   - Check your original project location")
    print("   - Look for files named 'prospect-data.csv'")
    print("   - Copy them to the appropriate directories")
    
    print("\n4. 🚀 Once data is in place:")
    print("   - Run: source .venv/bin/activate")
    print("   - Run: jupyter notebook notebooks/")
    print("   - Open and run your EDA notebook")
    
    print("\n5. 📊 Expected data structure:")
    print("   data/")
    print("   ├── raw/")
    print("   │   └── prospect-data.csv")
    print("   └── processed/")
    print("       ├── reports_with_bert_embeddings.csv")
    print("       └── reports_with_embeddings.csv")

def main():
    """Main function."""
    print("🚀 Setting up data directory structure...\n")
    
    setup_data_structure()
    print_instructions()
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)

if __name__ == "__main__":
    main() 