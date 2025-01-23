import config
from utils import make_dirs, load_data
from preprocess import preprocess_data, save_data
from Pipeline.feature_extraction import create_tfidf_matrix
from Pipeline.classifier_pipeline import create_pipeline, create_optimized_pipeline
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
        
        # Create and train the pipeline depending on config.py
        if config.TRAIN_MODEL:
            pipeline = create_pipeline()
            X = data.to_dict(orient='records')
            y = data['label'].tolist()
            pipeline.fit(X, y)
            joblib.dump(pipeline, config.DEFAULT_PIPELINE_PATH)
            print("Pipeline trained and saved.")

        if config.TRAIN_OPTIMIZED_MODEL:
            if config.OPTIMIZATION == 5:
                for opt_type in range(1, 5):
                    pipeline, type_name = create_optimized_pipeline(config.PARAM_GRID, opt_type)
                    pipeline.fit(X, y)
                    os.makedirs(config.PIPELINE_PATH, exist_ok=True)
                    joblib.dump(pipeline, os.path.join(config.PIPELINE_PATH, 'pipeline_' + type_name + '.pkl'))
                    print(f"{type_name} optimized pipeline trained and saved.")
                    print(f"{type_name} best parameters: {pipeline.best_params_}")
            else:
                pipeline, type_name = create_optimized_pipeline(config.PARAM_GRID, config.OPTIMIZATION)
                pipeline.fit(X, y)
                joblib.dump(pipeline, os.path.join(config.PIPELINE_PATH, 'pipeline_' + type_name + '.pkl'))
                print(f"{type_name} optimized pipeline trained and saved.")
                print(f"{type_name} best parameters: {pipeline.best_params_}")
        
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()