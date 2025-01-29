import os

# Preprocessing
PREPROCESS = False # Preprocess the data?
NUMBER_OF_ROWS = 500 # Limit number of rows to preprocess (0 for all rows)
TWO_COLUMNS = True # Combine two columns into one? Set False if dataset has only one column and label
DATA_COLUMN_NAME_1 = 'title' # Data column name for the first column
DATA_COLUMN_NAME_2 = 'text' # Data column name for the second column
DATA_LABEL = 'label' # Data column name for the label

# Model training
TRAIN_MODEL = False  # Train the model?
TRAIN_OPTIMIZED_MODEL = False # Train the optimized model?
# Optimization methods:
# 1: GridSearchCV
# 2: HalvingGridSearchCV
# 3: RandomizedSearchCV
# 4: HalvingRandomSearchCV
# 5: ALL (run all the above methods)
OPTIMIZATION = 1

CREATE_HEATMAP = True # Create the heatmap for the optimized model? (Requires TRAIN_OPTIMIZED_MODEL to be True)
HEATMAP_H = 6 # Height of the heatmap (1 = 100px)
HEATMAP_W = 6 # Width of the heatmap (1 = 100px)

PLOT_LEARNING_CURVE = True # Plot the learning curve for the optimized model? (Requires TRAIN_OPTIMIZED_MODEL to be True or trained PKT files)
PLOT_TEST_LINES = True # Plot the test lines in the learning curve?

CROSS_VALIDATIONS = 4 # Number of cross-validations  
NJOBS = -1 # Number of jobs to run in parallel. -1 = all processors.
VERBOSE = 1 # Verbosity level for the optimization methods (0, 1, 2, 3)

# Model parameters (for consistency)
RANDOM_STATE = 32

# Parameter grid for GridSearchCV
PARAM_GRID = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__gamma": [0.001, 0.01, 0.1, 1],
    "classifier__kernel": ["poly", "rbf", "sigmoid"],
    "classifier__degree": [2, 4, 5],  
    "classifier__coef0": [0.0, 0.5, 1.0]  
}

# Config for filepaths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(BASE_DIR, DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, DATA_DIR, 'processed')
OUTPUTS = os.path.join(BASE_DIR, 'output')

# Filenames
RAW_DATA_FILENAME = 'fake-news.csv'
PROCESSED_DATA_FILENAME = 'preprocessed_data.json'
PROCESSED_DATA_PKL = 'preprocessed_data.pkl'
SCORES = 'scores.json'
LEARNING_CURVE = 'learning_curve.png'

# Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, RAW_DATA_DIR, RAW_DATA_FILENAME)
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME)
SCORES_PATH = os.path.join(OUTPUTS, SCORES)
HEATMAP_DIR = os.path.join(OUTPUTS, 'heatmap')
LEARNING_CURVE_DIR = os.path.join(OUTPUTS, 'learning_curve')
LEARNING_CURVE_PATH = os.path.join(LEARNING_CURVE_DIR, LEARNING_CURVE)
PROCESSED_DATA_PKL_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_DATA_PKL)

#Pipelines
PIPELINE_DIR = os.path.join(OUTPUTS, 'pipeline')
DEFAULT_PIPELINE_DIR = os.path.join(OUTPUTS, PIPELINE_DIR, 'pipeline.pkl')
