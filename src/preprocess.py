from trankit import Pipeline
import pandas as pd
import json
import config

p = Pipeline('english')

def preprocess_data(data, number_of_rows):
    if number_of_rows > 0:
        data = data.head(number_of_rows)

    print(f"Preprocessing {len(data)} documents...")    
    
    # Combine columns into a single text column depending on config.py
    if config.TWO_COLUMNS:
        data.loc[:, 'combined_text'] = data[config.DATA_COLUMN_NAME_1] + ' ' + data[config.DATA_COLUMN_NAME_2]
        data.drop(columns=[config.DATA_COLUMN_NAME_1, config.DATA_COLUMN_NAME_2], inplace=True)
    else:
        data.loc[:, 'combined_text'] = data[config.DATA_COLUMN_NAME_1]
        data.drop(columns=[config.DATA_COLUMN_NAME_1], inplace=True)

    data.loc[:, 'combined_text'] = data['combined_text'].str.lower()

    preprocessed_data = []
    total_docs = len(data['combined_text'])
    progress_interval = max(1, total_docs // 20)  # 5% of total documents

    # Note: Computationally expensive as it processes at token level
    for idx, doc in enumerate(data['combined_text']):
        processed_doc = p(doc)
        sentences = []
        for sentence in processed_doc['sentences']:
            tokens = []
            for token in sentence['tokens']:
                tokens.append({
                    'token_id': token['id'],
                    'token_text': token['text'],
                    'upos': token.get('upos', ''),
                    'xpos': token.get('xpos', ''),
                    'feats': token.get('feats', ''),
                    'head': token.get('head', -1),
                    'deprel': token.get('deprel', ''),
                    'lemma': token.get('lemma', ''),
                    'ner': token.get('ner', 'O'),
                    'token_dspan': token.get('dspan', (0, 0)),
                    'token_span': token.get('span', (0, 0))
                })
            sentences.append({
                'sentence_id': sentence['id'],
                'sentence_text': sentence['text'],
                'tokens': tokens
            })
        preprocessed_data.append({
            'doc_index': idx,
            'doc_text': processed_doc['text'],
            'label': data[config.DATA_LABEL].iloc[idx],
            'sentences': sentences
        })
        if (idx + 1) % progress_interval == 0 or (idx + 1) == total_docs:
            print(f"Processed {idx + 1}/{total_docs} documents")

    # Convert preprocessed data back to DataFrame
    preprocessed_df = pd.DataFrame(preprocessed_data)
    return preprocessed_df

def save_data(data, filepath):
    try:
        data_list = data.to_dict(orient='records')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"Preprocessed data saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        raise