import config
from utils import make_dirs, load_data, save_scores, extract_best_fold_scores
from preprocess import preprocess_data, save_data
from Pipeline.classifier_pipeline import create_pipeline, create_optimized_pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import joblib
import os

def main():
    # Make directories and load csv data
    make_dirs()
    data = load_data(config.RAW_DATA_PATH)
    if data is not None:
        # Preprocess depending on config.py
        if config.PREPROCESS:
            data = preprocess_data(data, config.NUMBER_OF_ROWS)
            save_data(data, config.PROCESSED_DATA_PATH)
        else:
            data = pd.read_json(config.PROCESSED_DATA_PATH, orient='records')
        
        # Load data and labels into X and y for training
        y = data['label'].tolist()
        X = data.drop(columns=['label'])
        X = X.to_dict(orient='records')

        # Create and train the pipeline depending on config.py
        if config.TRAIN_MODEL:
            pipeline = create_pipeline()
            scores = cross_val_score(pipeline, X, y, cv=config.CROSS_VALIDATIONS, n_jobs=config.NJOBS)
            pipeline.fit(X, y)
            print(f"Cross-validation score: {scores}")

            classifier_params = {key: value for key, value in pipeline.get_params().items() if 'classifier__' in key}

            save_scores(scores, "default_pipeline", config.SCORES_PATH, classifier_params) 
            joblib.dump(pipeline, config.DEFAULT_PIPELINE_PATH)
            print("Pipeline trained and saved.")

        if config.TRAIN_OPTIMIZED_MODEL:
            if config.OPTIMIZATION == 5:
                for opt_type in range(1, 5):
                    pipeline, type_name, time = create_optimized_pipeline(config.PARAM_GRID, opt_type, X, y)
                    os.makedirs(config.PIPELINE_PATH, exist_ok=True)
                    joblib.dump(pipeline, os.path.join(config.PIPELINE_PATH, 'pipeline_' + type_name + '.pkl'))
                    
                    fold_scores, parameters, total_fits = extract_best_fold_scores(pipeline)
                    save_scores(fold_scores, type_name, config.SCORES_PATH, parameters, time, total_fits)
                    print(f"{type_name} best parameters: {pipeline.best_params_}")
            else:
                pipeline, type_name, time = create_optimized_pipeline(config.PARAM_GRID, config.OPTIMIZATION, X, y)
                joblib.dump(pipeline, os.path.join(config.PIPELINE_PATH, 'pipeline_' + type_name + '.pkl'))

                fold_scores, parameters, total_fits = extract_best_fold_scores(pipeline)
                save_scores(fold_scores, type_name, config.SCORES_PATH, parameters, time, total_fits)

                print(f"{type_name} optimized pipeline trained and saved.")
                print(f"{type_name} best parameters: {pipeline.best_params_}")
                print (f"{type_name} best score: {pipeline.best_score_}")
        
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()