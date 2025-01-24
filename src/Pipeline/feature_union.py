from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from Pipeline.transformers import TokenLevelFeatureExtractor, DocumentPreprocessor

def create_feature_union():
    feature_union = FeatureUnion([
        ('tfidf', Pipeline([
            ('preprocess', DocumentPreprocessor()),
            ('tfidf', TfidfVectorizer())
        ])),
        ('ner', Pipeline([
            ('extract', TokenLevelFeatureExtractor('ner')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('upos', Pipeline([
            ('extract', TokenLevelFeatureExtractor('upos')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('xpos', Pipeline([
            ('extract', TokenLevelFeatureExtractor('xpos')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('lemma', Pipeline([
            ('extract', TokenLevelFeatureExtractor('lemma')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]))
    ])   
    return feature_union