from trankit import Pipeline
import pandas as pd

p = Pipeline('english')

def preprocess_data(data):
    # Combine title and text into a single column
    data['combined_text'] = data['title'] + ' ' + data['text']
    data.drop(columns=['title', 'text'], inplace=True)

    data['combined_text'] = data['combined_text'].str.lower()

    data = data.head(100)

    preprocessed_data = []
    total_docs = len(data['combined_text'])
    progress_interval = max(1, total_docs // 20)  # 5% of total documents

    for idx, doc in enumerate(data['combined_text']):
        processed_doc = p(doc)
        for sentence in processed_doc['sentences']:
            for token in sentence['tokens']:
                preprocessed_data.append({
                    'doc_index': idx,
                    'doc_text': processed_doc['text'],
                    'sentence_id': sentence['id'],
                    'sentence_text': sentence['text'],
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
        if (idx + 1) % progress_interval == 0 or (idx + 1) == total_docs:
            print(f"Processed {idx + 1}/{total_docs} documents")

    # Convert preprocessed data back to DataFrame
    preprocessed_df = pd.DataFrame(preprocessed_data)
    return preprocessed_df

def save_data(data, filepath):
    data.to_csv(filepath, sep=';', index=False)
    print(f"Preprocessed data saved to {filepath}")
