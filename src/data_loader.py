import pandas as pd

def load_data(filepath):
    try:
        data = pd.read_csv(filepath, delimiter=';')
        return data
    except pd.errors.ParserError as e:
        print(f"Error parsing {filepath}: {e}")
        return None