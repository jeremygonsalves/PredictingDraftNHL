"""
Visualization utilities for the NHL Draft Prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from ..config.settings import FIGURES_DIR, FIGURE_SIZE, DPI, COLOR_PALETTE, HOCKEY_POSITIONS


def setup_plotting_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use('default')
    sns.set_palette(COLOR_PALETTE)
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI


def plot_drafted_vs_ranking(data, title="", save_path=None):
    """
    Plot drafted position vs average ranking with reference line.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing 'Drafted' and 'Average Ranking' columns
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the figure
    """
    if data.empty:
        print(f"No data available for {title}")
        return
    
    # Filter out missing values
    valid_data = data.dropna(subset=['Drafted', 'Average Ranking'])
    
    if valid_data.empty:
        print(f"No valid data for plotting {title}")
        return
    
    # Create the plot
    plt.figure(figsize=FIGURE_SIZE)
    
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
    
    plt.show()
    
    # Calculate correlation
    correlation = valid_data['Drafted'].corr(valid_data['Average Ranking'])
    print(f"Correlation between Drafted and Average Ranking: {correlation:.3f}")
    
    return correlation


def plot_position_distribution(data, save_path=None):
    """
    Plot the distribution of player positions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing 'Position' column
    save_path : str or Path, optional
        Path to save the figure
    """
    plt.figure(figsize=FIGURE_SIZE)
    position_counts = data['Position'].value_counts()
    position_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Player Positions')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
    
    plt.show()
    
    print("Position distribution:")
    print(position_counts)
    
    return position_counts


def plot_team_distribution(data, save_path=None):
    """
    Plot the distribution of teams and number of players drafted.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing 'Team' column
    save_path : str or Path, optional
        Path to save the figure
    """
    plt.figure(figsize=(15, 8))
    team_counts = data['Team'].value_counts()
    team_counts.plot(kind='bar', color='lightcoral')
    plt.title('Distribution of Teams and Number of Players Drafted')
    plt.xlabel('Teams')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
    
    plt.show()
    
    print("Top 10 teams by draft picks:")
    print(team_counts.head(10))
    
    return team_counts


def plot_draft_position_distribution(data, save_path=None):
    """
    Plot the distribution of draft positions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing 'Drafted' column
    save_path : str or Path, optional
        Path to save the figure
    """
    plt.figure(figsize=FIGURE_SIZE)
    draft_counts = data.groupby('Drafted').size()
    plt.bar(draft_counts.index, draft_counts.values, color='lightgreen', alpha=0.7)
    plt.title('Distribution of Draft Positions')
    plt.xlabel('Draft Position')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
    
    plt.show()
    
    print(f"Draft position statistics:")
    print(f"Mean draft position: {data['Drafted'].mean():.1f}")
    print(f"Median draft position: {data['Drafted'].median():.1f}")
    print(f"Min draft position: {data['Drafted'].min()}")
    print(f"Max draft position: {data['Drafted'].max()}")
    
    return draft_counts


def plot_token_distribution(token_analysis, scouting_report_cols, bert_model, save_path=None):
    """
    Plot the distribution of token counts in scouting reports.
    
    Parameters
    ----------
    token_analysis : pd.DataFrame
        DataFrame with token count columns
    scouting_report_cols : list
        List of scouting report column names
    bert_model : SentenceTransformer
        BERT model to get max sequence length
    save_path : str or Path, optional
        Path to save the figure
    """
    max_seq_length = bert_model.max_seq_length
    
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(scouting_report_cols):
        if idx >= len(axes):
            break
            
        token_counts = token_analysis[f'{col}_tokens'].dropna()
        
        if not token_counts.empty:
            axes[idx].hist(token_counts, bins=30, alpha=0.7, color='skyblue')
            axes[idx].axvline(max_seq_length, color='red', linestyle='--', 
                             label=f'BERT max length ({max_seq_length})')
            axes[idx].set_title(f'{col}\\n(Mean: {token_counts.mean():.1f})')
            axes[idx].set_xlabel('Token Count')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', 
                          transform=axes[idx].transAxes)
            axes[idx].set_title(col)
    
    # Hide unused subplots
    for idx in range(len(scouting_report_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Distribution of Token Counts in Scouting Reports", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
    
    plt.show()


def plot_model_comparison(summary_data, save_path=None):
    """
    Plot model performance comparison.
    
    Parameters
    ----------
    summary_data : list
        List of dictionaries containing model performance metrics
    save_path : str or Path, optional
        Path to save the figure
    """
    if len(summary_data) == 0 or 'Error' in summary_data[0].values():
        print("No valid model results to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    models = [row['Model'] for row in summary_data if row['Accuracy'] != 'Error']
    accuracies = [row['Accuracy'] for row in summary_data if row['Accuracy'] != 'Error']
    plt.bar(models, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # F1 Score comparison
    plt.subplot(2, 2, 2)
    f1_scores = [row['F1 Score'] for row in summary_data if row['F1 Score'] != 'Error']
    plt.bar(models, f1_scores, color='lightcoral')
    plt.title('Model F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    
    # CV Accuracy comparison
    plt.subplot(2, 2, 3)
    cv_accuracies = [row['CV Accuracy'] for row in summary_data if row['CV Accuracy'] != 'Error']
    cv_stds = [row['CV Std'] for row in summary_data if row['CV Std'] != 'Error']
    plt.bar(models, cv_accuracies, yerr=cv_stds, color='lightgreen', capsize=5)
    plt.title('Cross-Validation Accuracy Comparison')
    plt.ylabel('CV Accuracy')
    plt.xticks(rotation=45)
    
    # Precision vs Recall
    plt.subplot(2, 2, 4)
    precisions = [row['Precision'] for row in summary_data if row['Precision'] != 'Error']
    recalls = [row['Recall'] for row in summary_data if row['Recall'] != 'Error']
    plt.scatter(precisions, recalls, s=100, alpha=0.7)
    for i, model in enumerate(models):
        plt.annotate(model, (precisions[i], recalls[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=DPI)
    
    plt.show()


def save_model_results(summary_df, filename="model_results.csv"):
    """
    Save model results to a CSV file.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame containing model performance results
    filename : str
        Name of the file to save
    """
    save_path = TABLES_DIR / filename
    summary_df.to_csv(save_path, index=False)
    print(f"Model results saved to {save_path}") 