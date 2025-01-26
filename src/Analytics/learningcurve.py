import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, learning_curve
import config

        

def plot_learning_curves():
    model_files = [f for f in os.listdir(config.PIPELINE_PATH) if f.endswith('.pkl')]
    
    if not model_files:
        print("No models found in the specified directory.")
        return
    
    X, y = joblib.load(os.path.join(config.PROCESSED_DATA_PKL_PATH))
    
    # Create a single figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_file in model_files:
        model_path = os.path.join(config.PIPELINE_PATH, model_file)
        model = joblib.load(model_path)
        
        train_sizes, train_scores, test_scores = learning_curve(
            model , X, y, cv=config.CROSS_VALIDATIONS, n_jobs=config.NJOBS, random_state=config.RANDOM_STATE
        )
        
        train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)
        
        ax.plot(train_sizes, train_scores_mean, 'o-', label=f"{os.path.splitext(model_file)[0]} Train")
        ax.plot(train_sizes, test_scores_mean, 'o-', label=f"{os.path.splitext(model_file)[0]} Test")

    # Add legend manually with model names
    ax.legend(loc="best")

    # Customize plot
    plt.title("Learning Curves for All Models")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()