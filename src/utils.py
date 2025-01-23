import os
import config
import pandas as pd

def make_dirs():
    #DATA
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    #OUTPUTS
    os.makedirs(config.OUTPUTS, exist_ok=True)
    os.makedirs(config.PIPELINE_PATH, exist_ok=True)

def load_data(filepath):
    try:
        data = pd.read_csv(filepath, delimiter=';')
        return data
    except pd.errors.ParserError as e:
        print(f"Error parsing {filepath}: {e}")
        return None