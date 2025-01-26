import os
import config
import pandas as pd
import json
import numpy as np
from collections import OrderedDict

def make_dirs():
    #DATA
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    #OUTPUTS
    os.makedirs(config.OUTPUTS, exist_ok=True)
    if config.TRAIN_MODEL or config.TRAIN_OPTIMIZED_MODEL:
        os.makedirs(config.PIPELINE_DIR, exist_ok=True)
    if config.CREATE_HEATMAP:
        os.makedirs(config.HEATMAP_DIR, exist_ok=True)
    if config.PLOT_LEARNING_CURVE:
        os.makedirs(config.LEARNING_CURVE_DIR, exist_ok=True)
    

def load_data(filepath):
    try:
        data = pd.read_csv(filepath, delimiter=';')
        return data
    except pd.errors.ParserError as e:
        print(f"Error parsing {filepath}: {e}")
        return None
    
def extract_best_fold_scores(pipeline):
    cv_results = pipeline.cv_results_
    best_index = pipeline.best_index_
    scores = []
    params = cv_results['params'][best_index]
    
    # Dynamically find all fold scores for the best index
    fold_keys = [key for key in cv_results.keys() if key.startswith('split') and key.endswith('_test_score')]
    for key in fold_keys:
        scores.append(round(cv_results[key][best_index], 3))

    # Extract only the classifier parameters
    classifier_params = {key: value for key, value in params.items() if 'classifier__' in key}

    n_splits = len(fold_keys)  # Number of splits (folds)
    n_candidates = len(cv_results['params'])  # Number of candidates
    total_fits = n_splits * n_candidates  

    return scores, classifier_params, total_fits
    
def save_scores(scores, name, filename, used_params, time_spent=None, total_fits=None):
    # Define the order of the parameters
    param_order = [
        "classifier__kernel",
        "classifier__gamma",
        "classifier__degree",
        "classifier__coef0",
        "classifier__C"
    ]
    
    # Filter and order the parameters
    filtered_params = OrderedDict((key, used_params[key]) for key in param_order if key in used_params)
    
    scores_data = {
        "name": name,
        "used_params": filtered_params  
    }

    flat_scores = []
    for i, score in enumerate(scores, start=1):
        if isinstance(score, (int, float)):  # Ensure score is numeric
            scores_data[f"fold{i}"] = round(score, 3)
            flat_scores.append(score)
        elif isinstance(score, (list, np.ndarray)):  # Handle sequences
            for j, sub_score in enumerate(score, start=1):
                scores_data[f"fold{i}_{j}"] = round(sub_score, 3)
                flat_scores.append(sub_score)
        else:
            scores_data[f"fold{i}"] = score  # Handle non-numeric scores

    if time_spent is not None:
        scores_data["time_spent"] = round(time_spent, 3)
    if total_fits is not None:
        scores_data["total_fits"] = total_fits
    scores_data["mean"] = round(np.mean(flat_scores), 3)
    scores_data["std_dev"] = round(np.std(flat_scores), 3)
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    for entry in existing_data:
        if entry["name"] == name:
            entry.update(scores_data)
            break
    else:
        existing_data.append(scores_data)

    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)