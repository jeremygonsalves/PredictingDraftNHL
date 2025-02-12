# NHL Draft Prediction Project

## Overview
This project explores historical NHL draft data to build predictive models for future player performance. It involves data collection, preprocessing, exploratory analysis, model construction, and evaluating draft outcomes.

## Main Steps
1. Gather data from reliable hockey statistics sources.
2. Clean and prepare datasets for modeling.
3. Experiment with multiple models (e.g., regression, classification).
4. Evaluate performance through validation metrics.
5. Present findings with visualizations and tables.

## Project Structure
| Directory | Description                                               |
|-----------|-----------------------------------------------------------|
| code/     | Contains scripts for data cleaning, feature engineering, and model training (e.g., data_cleaner.py, feature_engineer.py) |
| data/     | Holds raw and processed datasets                          |
| images/   | Stores visual resources and figures                       |
| test/     | Includes test data and scripts for evaluation             |
| train/    | Contains training data and relevant scripts               |


| File              | Functionality                                          |
|-------------------|--------------------------------------------------------|
| data_cleaner.py   | Preprocesses raw data (e.g., removes duplicates)       |
| feature_engineer.py | Generates advanced features (e.g., points-per-game) |
| model_trainer.py  | Trains and evaluates models                            |
| analysis.ipynb    | Combines steps for quick analysis and visualization    |

## Results
| Metric          | Value      |
|-----------------|------------|
| Accuracy        | ~80%       |
| Precision       | ~0.78      |
| Recall          | ~0.75      |
| F1 Score        | ~0.76      |

## Next Steps
- Refine features and hyperparameters
- Explore ensemble methods
- Monitor real-world performance after each draft cycle
