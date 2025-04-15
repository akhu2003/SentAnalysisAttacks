import pandas as pd
from langdetect import detect
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from joblib import Parallel, delayed

print('reading csv')
df = pd.read_csv('mbsa.csv',index_col= None).head(3000000)
print('done reading')

# Example function for language detection with error handling
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return None

# Assuming df is your DataFrame and 'text' is your column
# Use ProcessPoolExecutor to parallelize language detection


lang_lst = Parallel(n_jobs=-1)(
    delayed(detect_language)(t) for t in tqdm(df['text'], total=len(df))
)

# Now, lang_lst contains the detected language for each tweet
df['Language'] = lang_lst
df.to_csv('btctweets.csv', sep=',', index=False, header=True, encoding='utf-8')

