import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TokenLevelFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for doc in X:
            doc_features = []
            for sentence in doc['sentences']:
                for token in sentence['tokens']:
                    doc_features.append(token[self.feature_name])
            # Join the token features into a single string
            features.append(' '.join(doc_features))
        return pd.Series(features).values.reshape(-1, 1)
    

class DocumentPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        documents = []
        for doc in X:
            doc_tokens = []
            for sentence in doc['sentences']:
                for token in sentence['tokens']:
                    doc_tokens.append(token['token_text'])
            documents.append(' '.join(doc_tokens))
        return documents