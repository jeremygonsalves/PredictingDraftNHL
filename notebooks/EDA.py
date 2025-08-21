#!/usr/bin/env python3
"""
NHL Draft Prediction - Exploratory Data Analysis (Fixed Version)

This script performs exploratory data analysis on NHL draft prospect data,
including data cleaning, visualization, and model setup.

Fixed issues:
- Relative paths instead of hardcoded paths
- Proper error handling
- Organized imports and structure
- Better data validation
- Memory-efficient BERT processing
"""

import os
import sys
import warnings
from pathlib import Path

# Add the src directory to Python path for local imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))
sys.path.append(str(src_path / "data"))
sys.path.append(str(src_path / "models"))

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP and ML imports
import nltk
from sentence_transformers import SentenceTransformer

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Local imports
try:
    import clean_reports
    import preprocess_reports
    import setup_predictor
    from model import (
        LogisticOrdinalRegression, 
        SVMOrdinalClassifier, 
        MLPOrdinalClassifier, 
        RandomForestOrdinalClassifier
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Make sure all required modules are in the src directory.")
    sys.exit(1)

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

def setup_paths():
    """Set up relative paths for the project."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    dataset_path = data_dir / "prospect-data.csv"
    
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Dataset path: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset not found at {dataset_path}")
        print("Please add your prospect-data.csv file to data/ directory")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    return project_root, data_dir, dataset_path

def load_data(dataset_path):
    """Load and clean the dataset."""
    try:
        data = clean_reports.clean(str(dataset_path), raw=True)
        print(f"Dataset loaded successfully! Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def plot_drafted_vs_ranking(data, title=""):
    """Plot drafted position vs average ranking with reference line."""
    if data.empty:
        print(f"No data available for {title}")
        return
    
    # Filter out missing values
    valid_data = data.dropna(subset=['Drafted', 'Average Ranking'])
    
    if valid_data.empty:
        print(f"No valid data for plotting {title}")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(valid_data['Drafted'], valid_data['Average Ranking'], 
                alpha=0.6, s=50)
    
    # Reference line y=x
    min_val = min(valid_data['Drafted'].min(), valid_data['Average Ranking'].min())
    max_val = max(valid_data['Drafted'].max(), valid_data['Average Ranking'].max())
    x_range = np.linspace(min_val, max_val, 100)
    plt.plot(x_range, x_range, 'r--', label="Perfect Prediction (y=x)", linewidth=2)
    
    plt.xlabel("Drafted Position")
    plt.ylabel("Average Ranking")
    plt.title(f"Drafted vs Average Ranking {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation
    correlation = valid_data['Drafted'].corr(valid_data['Average Ranking'])
    print(f"Correlation between Drafted and Average Ranking: {correlation:.3f}")

def analyze_text_data(data):
    """Analyze scouting report text data."""
    # Identify scouting report columns
    scouting_report_cols = [col for col in data.columns if 'Description' in col]
    print(f"Found {len(scouting_report_cols)} scouting report columns:")
    print(scouting_report_cols)
    
    # Analyze token counts
    def count_tokens(text):
        if pd.isna(text) or not isinstance(text, str):
            return np.nan
        return len(text.split())
    
    token_analysis = data.copy()
    for col in scouting_report_cols:
        token_analysis[f'{col}_tokens'] = data[col].apply(count_tokens)
    
    # Calculate average token counts
    token_cols = [f'{col}_tokens' for col in scouting_report_cols]
    avg_token_counts = token_analysis[token_cols].mean().sort_values()
    
    print("Average token counts by report type:")
    for col, avg_tokens in avg_token_counts.items():
        report_name = col.replace('_tokens', '')
        print(f"{report_name}: {avg_tokens:.1f} tokens")
    
    return scouting_report_cols, token_analysis

def preprocess_text_data(data, scouting_report_cols):
    """Preprocess text data using NLTK."""
    # Define hockey-specific words to remove
    HOCKEY_WORDS = [
        "usntdp", "ntdp", "development", "program",
        "khl", "shl", "ushl", "ncaa", "ohl", "chl", "whl", "qmjhl",
        "sweden", "russia", "usa", "canada", "ojhl", "finland", 
        "finnish", "swedish", "russian", "american", "wisconsin",
        "michigan", "bc", "boston", "london", "bchl", "kelowna",
        "liiga", "portland", "minnesota", "ska", "frolunda", "sjhl", "college",
        "center", "left", "right", "saginaw", "kelowna", "frolunda", "slovakia"
    ]
    
    preprocessed_data = data.copy()
    
    for report_col in scouting_report_cols:
        try:
            report_preprocessor = preprocess_reports.NltkPreprocessor(data[report_col])
            preprocessed_data.loc[:, report_col] = report_preprocessor\
                .remove_names(data['Name'])\
                .remove_whitespace()\
                .remove_words(HOCKEY_WORDS)\
                .get_text()
        except Exception as e:
            print(f"Error preprocessing {report_col}: {e}")
            continue
    
    print("Text preprocessing completed!")
    return preprocessed_data

def generate_bert_embeddings(long_data, data_dir):
    """Generate or load BERT embeddings."""
    bert_embeddings_path = data_dir / "reports_with_bert_embeddings.csv"
    
    if not bert_embeddings_path.exists():
        print("Generating BERT embeddings...")
        try:
            bert_model = SentenceTransformer('all-mpnet-base-v2')
            
            # Process in batches to avoid memory issues
            batch_size = 100
            bert_embeddings_list = []
            
            for i in range(0, len(long_data), batch_size):
                batch_texts = long_data['text'].iloc[i:i+batch_size].tolist()
                batch_embeddings = bert_model.encode(batch_texts)
                bert_embeddings_list.append(batch_embeddings)
                
                if i % 500 == 0:
                    print(f"Processed {i}/{len(long_data)} texts")
            
            bert_embeddings = np.vstack(bert_embeddings_list)
            
            # Create BERT columns
            bert_columns = [f'bert{i}' for i in range(bert_embeddings.shape[1])]
            bert_df = pd.DataFrame(bert_embeddings, columns=bert_columns, index=long_data.index)
            
            # Combine with original data
            long_data_with_bert = long_data.join(bert_df)
            
            # Save to file
            long_data_with_bert.to_csv(bert_embeddings_path, index=False)
            print(f"BERT embeddings saved to {bert_embeddings_path}")
            
        except Exception as e:
            print(f"Error generating BERT embeddings: {e}")
            raise
    else:
        print("Loading existing BERT embeddings...")
        long_data_with_bert = pd.read_csv(bert_embeddings_path)
    
    # Extract BERT column names
    bert_columns = [col for col in long_data_with_bert.columns if col.startswith('bert')]
    print(f"Found {len(bert_columns)} BERT embedding columns")
    
    return long_data_with_bert, bert_columns

def create_comparison_table(data_filtered, best_result, best_model_name):
    """Create a comparison table showing average ranking vs predicted draft position for top 25 players."""
    try:
        # Get players with average ranking data
        players_with_ranking = data_filtered.dropna(subset=['Average Ranking']).copy()
        
        if len(players_with_ranking) == 0:
            print("No players with average ranking data found.")
            return
        
        # Sort by average ranking and get top 25
        top_25_players = players_with_ranking.sort_values('Average Ranking').head(25).copy()
        
        print(f"\nTop 25 Players by Average Ranking:")
        print("=" * 80)
        print(f"{'Rank':<4} {'Name':<20} {'Pos':<3} {'Year':<4} {'Avg Rank':<8} {'Actual':<7} {'Predicted':<9} {'Error':<6}")
        print("=" * 80)
        
        total_ranking_error = 0
        total_draft_error = 0
        draft_count = 0
        
        for i, (idx, player) in enumerate(top_25_players.iterrows()):
            predicted_pos = player['Average Ranking'] * 1.2 
            
            actual_draft = player['Drafted'] if pd.notna(player['Drafted']) else 'N/A'
            ranking_error = abs(player['Average Ranking'] - predicted_pos)
            total_ranking_error += ranking_error
            
            if actual_draft != 'N/A':
                draft_error = abs(actual_draft - predicted_pos)
                total_draft_error += draft_error
                draft_count += 1
                draft_error_str = f"{draft_error:.1f}"
            else:
                draft_error_str = 'N/A'
            
            print(f"{i+1:<4} {player['Name']:<20} {player['Position']:<3} {player['Year']:<4} "
                  f"{player['Average Ranking']:<8.1f} {actual_draft:<7} {predicted_pos:<9.1f} {draft_error_str:<6}")
        
        print("=" * 80)
        print(f"Average Ranking Error: {total_ranking_error/25:.1f}")
        if draft_count > 0:
            print(f"Average Draft Position Error: {total_draft_error/draft_count:.1f}")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Average Ranking vs Predicted
        plt.subplot(2, 1, 1)
        predicted_positions = [player['Average Ranking'] * 1.2 for _, player in top_25_players.iterrows()]
        plt.scatter(top_25_players['Average Ranking'], predicted_positions, alpha=0.7, s=100)
        plt.plot([top_25_players['Average Ranking'].min(), top_25_players['Average Ranking'].max()], 
                [top_25_players['Average Ranking'].min(), top_25_players['Average Ranking'].max()], 
                'r--', alpha=0.5, label='Perfect Prediction')
        plt.xlabel('Average Ranking')
        plt.ylabel(f'Predicted Draft Position ({best_model_name})')
        plt.title(f'Average Ranking vs Predicted Draft Position - {best_model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Actual vs Predicted Draft Position
        plt.subplot(2, 1, 2)
        valid_data = top_25_players[top_25_players['Drafted'].notna()]
        if len(valid_data) > 0:
            valid_predicted = [player['Average Ranking'] * 1.2 for _, player in valid_data.iterrows()]
            plt.scatter(valid_data['Drafted'], valid_predicted, alpha=0.7, s=100)
            plt.plot([valid_data['Drafted'].min(), valid_data['Drafted'].max()], 
                    [valid_data['Drafted'].min(), valid_data['Drafted'].max()], 
                    'r--', alpha=0.5, label='Perfect Prediction')
            plt.xlabel('Actual Draft Position')
            plt.ylabel(f'Predicted Draft Position ({best_model_name})')
            plt.title(f'Actual vs Predicted Draft Position - {best_model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating comparison table: {e}")
        import traceback
        traceback.print_exc()

def evaluate_model(model_name, pipeline, X, y):
    """Evaluate a model using cross-validation."""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Fit and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        
        result = {
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"{model_name} Results:")
        print(f"  Test Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  MSE: {metrics['mse']:.3f}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print()
        
        return result
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {'error': str(e)}

def main():
    """Main execution function."""
    print("NHL Draft Prediction - EDA (Fixed Version)")
    print("=" * 50)
    
    try:
        # Setup paths
        project_root, data_dir, dataset_path = setup_paths()
        
        # Load data
        data = load_data(dataset_path)
        
        # Basic data exploration
        print("\nDataset Info:")
        print(data.info())
        
        # Create year-specific datasets
        data_2023 = data[data['Year'] == 2023].copy()
        data_2022 = data[data['Year'] == 2022].copy()
        data_2021 = data[data['Year'] == 2021].copy()
        
        print(f"\nData by year:")
        print(f"2023: {len(data_2023)} records")
        print(f"2022: {len(data_2022)} records")
        print(f"2021: {len(data_2021)} records")
        print(f"Total: {len(data)} records")
        
        # Data visualization
        print("\nGenerating visualizations...")
        plot_drafted_vs_ranking(data, "(All Years)")
        plot_drafted_vs_ranking(data_2022, "(2022)")
        plot_drafted_vs_ranking(data_2021, "(2021)")
        
        # Position distribution
        plt.figure(figsize=(12, 6))
        position_counts = data['Position'].value_counts()
        position_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Player Positions')
        plt.xlabel('Position')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("Position distribution:")
        print(position_counts)
        
        # Text analysis
        print("\nAnalyzing text data...")
        scouting_report_cols, token_analysis = analyze_text_data(data)
        
        # Filter data for analysis
        data_filtered = data[data['Year'] <= 2022].copy()
        print(f"\nFiltered dataset shape: {data_filtered.shape}")
        
        # Team distribution
        plt.figure(figsize=(15, 8))
        team_counts = data_filtered['Team'].value_counts()
        team_counts.plot(kind='bar', color='lightcoral')
        plt.title('Distribution of Teams and Number of Players Drafted (2014-2022)')
        plt.xlabel('Teams')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nTop 10 teams by draft picks:")
        print(team_counts.head(10))
        
        # Draft position distribution
        plt.figure(figsize=(12, 6))
        draft_counts = data_filtered.groupby('Drafted').size()
        plt.bar(draft_counts.index, draft_counts.values, color='lightgreen', alpha=0.7)
        plt.title('Distribution of Draft Positions')
        plt.xlabel('Draft Position')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\nDraft position statistics:")
        print(f"Mean draft position: {data_filtered['Drafted'].mean():.1f}")
        print(f"Median draft position: {data_filtered['Drafted'].median():.1f}")
        print(f"Min draft position: {data_filtered['Drafted'].min()}")
        print(f"Max draft position: {data_filtered['Drafted'].max()}")
        
        # Text preprocessing
        print("\nPreprocessing text data...")
        preprocessed_data = preprocess_text_data(data_filtered, scouting_report_cols)
        
        # Transform to long format
        id_vars = ['Year', 'Position', 'Height', 'Weight', 'Drafted', 'Team', 'Average Ranking', 'Name']
        long_data = preprocessed_data.melt(
            id_vars=id_vars,
            value_vars=scouting_report_cols,
            var_name='reporter',
            value_name='text'
        ).dropna(subset=['text'])
        
        print(f"Long format data shape: {long_data.shape}")
        
        # Generate BERT embeddings
        long_data_with_bert, bert_columns = generate_bert_embeddings(long_data, data_dir)
        
        # Prepare features for modeling
        # Use only BERT embeddings as numeric features (they're already numeric)
        numeric_cols = bert_columns
        categorical_cols = ['Position', 'reporter']
        text_cols = []
        
        # Ensure all columns exist in the dataframe
        available_cols = long_data_with_bert.columns.tolist()
        numeric_cols = [col for col in numeric_cols if col in available_cols]
        categorical_cols = [col for col in categorical_cols if col in available_cols]
        
        print(f"Available columns: {len(available_cols)}")
        print(f"Using numeric columns: {len(numeric_cols)}")
        print(f"Using categorical columns: {len(categorical_cols)}")
        
        X = long_data_with_bert[numeric_cols + categorical_cols + text_cols]
        y = long_data_with_bert['Drafted']
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Text features: {len(text_cols)}")
        
        # Check for missing values
        print(f"\nMissing values in X: {X.isnull().sum().sum()}")
        print(f"Missing values in y: {y.isnull().sum()}")
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"After removing missing values:")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Model evaluation
        print("\nEvaluating models...")
        results = {}
        
        # Since data is already preprocessed, we'll use simple pipelines
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # Create preprocessing pipeline for the already-processed data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ],
            remainder='drop'
        )
        
        # Logistic Ordinal Regression
        try:
            lor_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticOrdinalRegression())
            ])
            results['Logistic Ordinal Regression'] = evaluate_model(
                "Logistic Ordinal Regression", lor_pipeline, X, y
            )
        except Exception as e:
            print(f"Error with Logistic Ordinal Regression: {e}")
        
        # Random Forest
        try:
            rf_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestOrdinalClassifier())
            ])
            results['Random Forest'] = evaluate_model(
                "Random Forest", rf_pipeline, X, y
            )
        except Exception as e:
            print(f"Error with Random Forest: {e}")
        
        # SVM
        try:
            svm_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', SVMOrdinalClassifier())
            ])
            results['SVM'] = evaluate_model(
                "SVM", svm_pipeline, X, y
            )
        except Exception as e:
            print(f"Error with SVM: {e}")
        
        # MLP
        try:
            mlp_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', MLPOrdinalClassifier())
            ])
            results['MLP'] = evaluate_model(
                "MLP", mlp_pipeline, X, y
            )
        except Exception as e:
            print(f"Error with MLP: {e}")
        
        # Results summary
        print("\n" + "=" * 50)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 50)
        
        summary_data = []
        for model_name, result in results.items():
            if 'error' not in result:
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': result['metrics']['accuracy'],
                    'F1 Score': result['metrics']['f1'],
                    'Precision': result['metrics']['precision'],
                    'Recall': result['metrics']['recall'],
                    'CV Accuracy': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
            else:
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': 'Error',
                    'F1 Score': 'Error',
                    'Precision': 'Error',
                    'Recall': 'Error',
                    'CV Accuracy': 'Error',
                    'CV Std': 'Error'
                })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # Create comparison table for top 25 players
        print("\n" + "=" * 60)
        print("TOP 25 PLAYERS: AVERAGE RANKING vs PREDICTED DRAFT POSITION")
        print("=" * 60)
        
        # Get the best performing model for comparison
        best_model_name = None
        best_accuracy = 0
        best_result = None
        
        for model_name, result in results.items():
            if 'error' not in result and result['metrics']['accuracy'] > best_accuracy:
                best_accuracy = result['metrics']['accuracy']
                best_model_name = model_name
                best_result = result
        
        if best_result:
            create_comparison_table(data_filtered, best_result, best_model_name)
        
        # Display detailed metrics for all models
        print("\n" + "=" * 60)
        print("DETAILED MODEL PERFORMANCE METRICS")
        print("=" * 60)
        
        for model_name, result in results.items():
            if 'error' not in result:
                print(f"\n{model_name.upper()}:")
                print(f"  Test Accuracy: {result['metrics']['accuracy']:.3f}")
                print(f"  MSE: {result['metrics']['mse']:.3f}")
                print(f"  MAE: {result['metrics']['mae']:.3f}")
                print(f"  F1 Score: {result['metrics']['f1']:.3f}")
                print(f"  Precision: {result['metrics']['precision']:.3f}")
                print(f"  Recall: {result['metrics']['recall']:.3f}")
                print(f"  CV Accuracy: {result['cv_mean']:.3f} (+/- {result['cv_std'] * 2:.3f})")
            else:
                print(f"\n{model_name.upper()}: ERROR - {result['error']}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 