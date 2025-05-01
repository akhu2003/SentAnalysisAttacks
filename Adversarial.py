import pandas as pd
import numpy as np
from typing import List, Union


    
def Attack(X_test,y_test, directional = 'Neg', density = .1, seed = 42):
    X_test = list(X_test['text'])
    y_test = list(y_test)
    n = len(X_test)
    x = int(n*density)
    total_slots = n + x + 1
    step = total_slots / (x + 1)

    path = directional+' Adversarial GPT Tweets.csv'
    attack = pd.read_csv(path).sample(frac=1,random_state=seed).reset_index(drop=True).iloc[:,0]
    if directional =='Neg':
        truth_value = -1
    else:
        truth_value = 1

    attacked_dataset = []
    attacked_mapping = []
    insert_positions = [round((i + 1) * step) - 1 for i in range(x)]
    attack_pos = 0
    default_pos = 0
    for idx in range(n + x):
        if idx in insert_positions:
            attacked_dataset.append(attack[attack_pos])
            attacked_mapping.append(truth_value)
            attack_pos+=1
        else:
            attacked_dataset.append(X_test[default_pos])
            attacked_mapping.append(y_test[default_pos])
            default_pos+=1
    return pd.DataFrame({'text':attacked_dataset}),attacked_mapping,insert_positions
