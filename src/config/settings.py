"""
Configuration settings for the NHL Draft Prediction project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
PROSPECT_DATA_FILE = RAW_DATA_DIR / "prospect-data.csv"
BERT_EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "reports_with_bert_embeddings.csv"
WORD2VEC_EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "reports_with_embeddings.csv"

# Model settings
DEFAULT_BERT_MODEL = 'all-mpnet-base-v2'
DEFAULT_TFIDF_PARAMS = {
    'analyzer': 'word',
    'max_df': 0.5,
    'min_df': 0.04,
    'ngram_range': (1, 3)
}

# Default feature columns
DEFAULT_NUMERIC_COLS = ['Height', 'Weight']
DEFAULT_CATEGORICAL_COLS = ['Position']
DEFAULT_TEXT_COLS = ['all_reports']

# Hockey-specific words to remove during preprocessing
HOCKEY_WORDS = [
    "usntdp", "ntdp", "development", "program",
    "khl", "shl", "ushl", "ncaa", "ohl", "chl", "whl", "qmjhl",
    "sweden", "russia", "usa", "canada", "ojhl", "finland", 
    "finnish", "swedish", "russian", "american", "wisconsin",
    "michigan", "bc", "boston", "london", "bchl", "kelowna",
    "liiga", "portland", "minnesota", "ska", "frolunda", "sjhl", "college",
    "center", "left", "right", "saginaw", "kelowna", "frolunda", "slovakia"
]

# Hockey positions mapping
HOCKEY_POSITIONS = {
    'C': 'Center',
    'D': 'Defender',
    'RW': 'Right Wing',
    'LW': 'Left Wing',
    'G': 'Goalie'
}

# Model evaluation settings
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = "husl"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, 
                  FIGURES_DIR, TABLES_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 