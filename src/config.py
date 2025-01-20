import os

# Config for filepaths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Filenames
RAW_DATA_FILENAME = 'fake-news.csv'
PROCESSED_DATA_FILENAME = 'preprocessed_data.csv'

# Paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data/raw', RAW_DATA_FILENAME)
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data/processed', PROCESSED_DATA_FILENAME)

# Preprocessing
PREPROCESS = True