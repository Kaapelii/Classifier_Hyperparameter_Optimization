from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from Pipeline.transformers import TokenLevelFeatureExtractor, DocumentPreprocessor

def create_feature_union():
    feature_union = FeatureUnion([
        ('tfidf', Pipeline([
            ('preprocess', DocumentPreprocessor()),
            ('tfidf', TfidfVectorizer())
        ])),
        ('ner', Pipeline([
            ('extract', TokenLevelFeatureExtractor('ner')),
            ('tfidf', TfidfVectorizer())
        ])),
        ('upos', Pipeline([
            ('extract', TokenLevelFeatureExtractor('upos')),
            ('tfidf', TfidfVectorizer())
        ])),
        ('xpos', Pipeline([
            ('extract', TokenLevelFeatureExtractor('xpos')),
            ('tfidf', TfidfVectorizer())
        ])),
        ('lemma', Pipeline([
            ('extract', TokenLevelFeatureExtractor('lemma')),
            ('tfidf', TfidfVectorizer())
        ]))
    ])   
    return feature_union