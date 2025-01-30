import os
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay
import joblib
import config  
import numpy as np

def plot_learning_curves():
    model_files = [f for f in os.listdir(config.PIPELINE_DIR) if f.endswith('.pkl')]
    
    if not model_files:
        print("No models found in the specified directory.")
        return
    
    X, y = joblib.load(os.path.join(config.PROCESSED_DATA_PKL_PATH))

    fig, ax = plt.subplots(figsize=(10, 8))
    model_styles = {
        "pipeline_GridSearchCV.pkl": {
            'color': 'darkgreen',
            'train_markevery': 25,
            'test_markevery': 30,
            'marker': 'o',
            'markersize': 6,
            'linestyle': '--',
            'linestyle_test': ':'
        },
        "pipeline_HalvingGridSearchCV.pkl": {
            'color': 'yellow',
            'train_markevery': 35,
            'test_markevery': 40,
            'marker': 's',
            'markersize': 6,
            'linestyle': (0, (3, 5, 1, 5, 1, 5)),
            'linestyle_test': (0, (3, 5, 1, 5))

        },
        "pipeline_RandomizedSearchCV.pkl": {
            'color': 'red',
            'train_markevery': 45,
            'test_markevery': 50,
            'marker': '^',
            'markersize': 6,
            'linestyle': '--',
            'linestyle_test': ':'
        },
        "pipeline_HalvingRandomSearchCV.pkl": {
            'color': 'magenta',
            'train_markevery': 55,
            'test_markevery': 60,
            'marker': 'D',
            'markersize': 6,
            'linestyle': (0, (3, 5, 1, 5, 1, 5)),
            'linestyle_test': (0, (3, 5, 1, 5))
            
        },
        "default": {
            'color': 'blue',
            'train_markevery': 15,
            'test_markevery': 20,
            'marker': 'x',
            'markersize': 6,
            'linestyle': '--',
            'linestyle_test': ':'
        }
    }
    
    for model_file in model_files:
        model_path = os.path.join(config.PIPELINE_DIR, model_file)
        model = joblib.load(model_path)
        
        # Determine the color for the model
        style = model_styles.get(model_file, model_styles["default"])

        if config.PLOT_TEST_LINES:
            score_type = "both"
        else:
            score_type = "train"
        
        # Plot the learning curve 
        display = LearningCurveDisplay.from_estimator(
            model, X, y, ax=ax, cv=config.CROSS_VALIDATIONS, n_jobs=config.NJOBS, std_display_style=None, score_type=score_type
        )
        
        # Apply colors to train and test lines
        for line, label in zip(display.lines_, ["train", "test"]):
          line.set_color(style['color'])
          line.set_linestyle(style['linestyle'] if label == "train" else style['linestyle_test'])
          line.set_marker(style['marker'])
          line.set_markersize(style['markersize'])
          line.set_markevery(style['train_markevery'] if label == "train" else style['test_markevery'])
          line.set_linewidth(3)
          line.set_alpha(0.4 if label == "test" else 1)
          line.set_label(f"{model_file} ({label})")

    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Learning Curves", fontsize=16)
    ax.set_xlabel("Training Examples", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    
    plt.savefig(config.LEARNING_CURVE_PATH)
    plt.close(fig)