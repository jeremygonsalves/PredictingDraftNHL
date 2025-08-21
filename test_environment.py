#!/usr/bin/env python3
"""
Test script to verify the environment is set up correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn imported successfully")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import keras
        print("✅ Keras imported successfully")
    except ImportError as e:
        print(f"❌ Keras import failed: {e}")
        return False
    
    return True

def test_bert_model():
    """Test BERT model loading."""
    print("\n🧪 Testing BERT model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
        print("✅ BERT model loaded successfully")
        
        # Test encoding
        test_text = "This is a test sentence."
        embeddings = model.encode([test_text])
        print(f"✅ BERT encoding successful - Shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"❌ BERT model test failed: {e}")
        return False

def test_data_access():
    """Test data file access."""
    print("\n🧪 Testing data access...")
    
    try:
        data_file = Path("data/raw/prospect-data.csv")
        if data_file.exists():
            print(f"✅ Data file found: {data_file}")
            return True
        else:
            print(f"⚠️  Data file not found: {data_file}")
            return False
    except Exception as e:
        print(f"❌ Data access test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting environment tests...\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test BERT model
    bert_ok = test_bert_model()
    
    # Test data access
    data_ok = test_data_access()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    if imports_ok and bert_ok and data_ok:
        print("🎉 All tests passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Run: source .venv/bin/activate")
        print("2. Run: jupyter notebook notebooks/")
        print("3. Open and run your EDA notebook")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 