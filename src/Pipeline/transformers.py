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
        print(f"TokenLevelFeatureExtractor ({self.feature_name}): {len(features)} samples")
        return features
    

class DocumentLevelTFIDFExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_matrix):
        self.tfidf_matrix = tfidf_matrix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"DocumentLevelTFIDFExtractor: {self.tfidf_matrix.shape[0]} samples")
        return self.tfidf_matrix
    

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
        print(f"DocumentPreprocessor: {len(documents)} samples")
        return documents