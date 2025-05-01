from flair.models import TextClassifier
from flair.data import Sentence
from typing import List, Union

def DeepLearningAnalyzer(tweets: Union[str, List[str]]):
    classifier = TextClassifier.load('sentiment')
    prediction = []
    confidence = []
    for tweet in tweets:
        s = Sentence(tweet)
        classifier.predict(s)
        prediction.append(s.labels[0].value)
        confidence.append(s.labels[0].score)
    return [1 if p == 'POSITIVE' else -1 for p in prediction],confidence