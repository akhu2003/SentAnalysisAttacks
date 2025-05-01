from transformers import pipeline
from typing import List, Union
def TransformerAnalyzer(tweets: Union[str, List[str]]):
    classifier = pipeline("sentiment-analysis")
    prediction = []
    confidence = []
    for tweet in tweets:
        result = classifier(tweet)
        prediction.append(result[0]['label'])
        confidence.append(result[0]['score'])
    return [1 if p == 'POSITIVE' else -1 for p in prediction],confidence