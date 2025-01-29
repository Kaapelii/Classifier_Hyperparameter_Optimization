from utils import make_dirs, load_data, save_scores, extract_best_fold_scores
from preprocess import preprocess_data, save_data
from Pipeline.classifier_pipeline import create_pipeline, create_optimized_pipeline
from sklearn.model_selection import cross_val_score
from Analytics.learningcurve import plot_learning_curves

import pandas as pd
import joblib
import os
import config

def main():
    make_dirs()
    data = load_data(config.RAW_DATA_PATH)
    if data is not None:
        if config.PREPROCESS:
            data = preprocess_data(data, config.NUMBER_OF_ROWS)
            save_data(data, config.PROCESSED_DATA_PATH)
        else:
            data = pd.read_json(config.PROCESSED_DATA_PATH, orient='records')
        
        # Load data and labels into X and y for training
        y = data['label'].tolist()
        X = data.drop(columns=['label'])
        X = X.to_dict(orient='records')

        if config.PLOT_LEARNING_CURVE:
            # Save X and y for later use
            joblib.dump((X, y), os.path.join(config.PROCESSED_DATA_PKL_PATH))

        # Create and train the pipeline depending on config.py
        if config.TRAIN_MODEL:
            pipeline = create_pipeline()
            scores = cross_val_score(pipeline, X, y, cv=config.CROSS_VALIDATIONS, n_jobs=config.NJOBS)
            pipeline.fit(X, y)
            print(f"Cross-validation score: {scores}")

            classifier_params = {key: value for key, value in pipeline.get_params().items() if 'classifier__' in key}

            save_scores(scores, "default_pipeline", config.SCORES_PATH, classifier_params) 
            joblib.dump(pipeline, config.DEFAULT_PIPELINE_DIR)
            print("Pipeline trained and saved.")

        if config.TRAIN_OPTIMIZED_MODEL:
            if config.OPTIMIZATION == 5:
                for opt_type in range(1, 5):
                    pipeline, type_name, time = create_optimized_pipeline(config.PARAM_GRID, opt_type, X, y)
                    fold_scores, parameters, total_fits = extract_best_fold_scores(pipeline)
                    save_scores(fold_scores, type_name, config.SCORES_PATH, parameters, time, total_fits)
            else:
                pipeline, type_name, time = create_optimized_pipeline(config.PARAM_GRID, config.OPTIMIZATION, X, y)
                fold_scores, parameters, total_fits = extract_best_fold_scores(pipeline)
                save_scores(fold_scores, type_name, config.SCORES_PATH, parameters, time, total_fits)

        if config.PLOT_LEARNING_CURVE:
            plot_learning_curves()
        
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()