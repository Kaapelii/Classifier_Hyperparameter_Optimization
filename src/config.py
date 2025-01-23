import os

# Config for filepaths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(BASE_DIR, DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, DATA_DIR, 'processed')
OUTPUTS = os.path.join(BASE_DIR, 'output')

# Filenames
RAW_DATA_FILENAME = 'fake-news.csv'
PROCESSED_DATA_FILENAME = 'preprocessed_data.json'

# Preprocessing
PREPROCESS = False
NUMBER_OF_ROWS = 100
TWO_COLUMNS = True
DATA_COLUMN_NAME_1 = 'title'
DATA_COLUMN_NAME_2 = 'text'

# Model training
TRAIN_MODEL = True  # Train the model?
TRAIN_OPTIMIZED_MODEL = True # Train the optimized model?

# Optimization methods:
# 1: GridSearchCV
# 2: HalvingGridSearchCV
# 3: RandomizedSearchCV
# 4: HalvingRandomSearchCV
# 5: ALL (run all the above methods)
OPTIMIZATION = 4  

# Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, RAW_DATA_DIR, RAW_DATA_FILENAME)
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME)

#Pipelines
PIPELINE_PATH = os.path.join(OUTPUTS, 'pipeline')
DEFAULT_PIPELINE_PATH = os.path.join(OUTPUTS, PIPELINE_PATH, 'pipeline.pkl')

#Classifier
RANDOM_STATE = 32

# Parameter grid for GridSearchCV
PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "classifier__kernel": ["poly", "rbf", "sigmoid"],
    "classifier__degree": [2, 3, 4, 5],  
    "classifier__coef0": [0.0, 0.1, 0.5, 1.0]  
}