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
    model_colors = {
        "default": "blue",  
        "pipeline_GridSearchCV.pkl": "darkgreen",
        "pipeline_HalvingGridSearchCV.pkl": "purple",
        "pipeline_RandomizedSearchCV.pkl": "red",
        "pipeline_HalvingRandomSearchCV.pkl": "magenta",
        }   
    
    for model_file in model_files:
        model_path = os.path.join(config.PIPELINE_DIR, model_file)
        model = joblib.load(model_path)
        
        # Determine the color for the model
        if model_file in model_colors:
            color = model_colors[model_file]
        else:
            color = model_colors["default"] 

        if config.PLOT_TEST_LINES:
            score_type = "both"
        else:
            score_type = "train"
        
        # Plot the learning curve 
        display = LearningCurveDisplay.from_estimator(
            model, X, y, ax=ax, cv=config.CROSS_VALIDATIONS, n_jobs=config.NJOBS, std_display_style=None, score_type=score_type, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Apply colors to train and test lines
        # Apply colors to train and test lines
        for line, label in zip(display.lines_, ["train", "test"]):
            line.set_color(color)
            line.set_linewidth(2.5)
            if label == "test":
                line.set_alpha(0.3)  # Set test line to be semi-transparent
            line.set_label(f"{model_file} ({label})")

    

    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Learning Curves", fontsize=16)
    ax.set_xlabel("Training Examples", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    
    plt.savefig(config.LEARNING_CURVE_PATH)
    plt.close(fig)