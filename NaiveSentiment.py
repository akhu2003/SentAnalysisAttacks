import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List, Union


    
def NaiveAnalyzer(tweets: Union[str, List[str]]):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)['compound'] for t in tweets]
    return scores
