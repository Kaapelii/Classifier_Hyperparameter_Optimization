import os
import config


def make_dirs():
    #DATA
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    #OUTPUTS
    os.makedirs(config.OUTPUTS, exist_ok=True)
    os.makedirs(config.PIPELINE_PATH, exist_ok=True)
