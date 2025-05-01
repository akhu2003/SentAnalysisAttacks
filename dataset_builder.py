import pandas as pd

def dataset_builder(total_samples,split,random_state_int):
    assert split[0]+split[1] == 1
    #print('Opening Data')
    data = pd.read_csv('dataset.csv')
    pos_Data = data[data['Sentiment'] == 'Positive']
    neg_Data = data[data['Sentiment'] == 'Negative']
    
    sampled_pos_data = pos_Data.sample(n = int(split[0]*total_samples),random_state=random_state_int, ignore_index=True)
    sampled_neg_data = neg_Data.sample(n =int(split[1]*total_samples),random_state=random_state_int, ignore_index=True)
    pos_neg_df = pd.concat([sampled_pos_data, sampled_neg_data], axis=0)
    return pos_neg_df.sample(frac=1,random_state=random_state_int).reset_index(drop=True)