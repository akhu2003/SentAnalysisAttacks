import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Union
def logistic_analyze(
    train_tweets: Union[str, List[str]],
    train_tweets_sentiment: List[int],
    test_tweets: Union[str,List[str]],
    random_state: int
):
   
    # 1. Split into train/test
    X_train = train_tweets
    y_train = train_tweets_sentiment
    X_test = test_tweets

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10_000,      # limit vocab size
            ngram_range=(1,2),        # unigrams + bigrams
            strip_accents="unicode",
            lowercase=True,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            solver="liblinear",       # good for small datasets
            C=1.0,                    # inverse regularization strength
            max_iter=1000,
            random_state=random_state
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    positive_confidence = proba[:, 1]
    return [1 if y=='Positive' else -1 for y in y_pred],positive_confidence
