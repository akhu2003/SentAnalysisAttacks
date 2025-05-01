from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from NaiveSentiment import NaiveAnalyzer
from LogisticAgent import logistic_analyze
from dataset_builder import dataset_builder
from DeepLearning import DeepLearningAnalyzer
from TransformerAgent import TransformerAnalyzer
from Adversarial import Attack
from Defense import check_dataset
def experiment(samples = 100,train_samples = 400,random_state_int = 42, split = [.5,.5],
               Attack_Status = False,Attack_Direction = 'Neg', attack_density = .1,
               Defend_Status = False, defense_tolerance = .4):

    data = dataset_builder(total_samples = samples+train_samples, random_state_int=random_state_int,split = split)

    experiment_data = {
        'true_labels':[],
        'attacked_labels':[],
        'attack_locs':[],
        'defend_locs':[],
        'defend_labels':[],
        'defend_conf':[],
        'naive_sentiments': [],
        'naive_sentiments_conf':[],
        'logistic_regression': [],
        'logistic_regression_conf':[],
        'deep_learning':[],
        'deep_learning_conf': [],
        'transformer':[],
        'transformer_conf':[]
    }
    if len(data[data['Sentiment'].isin(['Positive','Negative'])]) != len(data):
        raise FileExistsError


    large_X = data[['tokenized no stop','text']]
    y = data['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(large_X,y, shuffle = True, test_size = samples, random_state = random_state_int)
    y_test = [1 if y == 'Positive' else -1 for y in y_test]
    experiment_data['true_labels'] = y_test
    if Attack_Status:
        #print('ATTACKED')
        mod_X_test,mod_y_test, adversarial_mappings = Attack(X_test=X_test,
                                                     y_test = y_test,
                                                     directional = Attack_Direction, 
                                                     density=attack_density,
                                                     seed = random_state_int)
        experiment_data['attacked_labels'] = mod_y_test
        experiment_data['attack_locs'] = adversarial_mappings
        X_test = mod_X_test
        y_test = mod_y_test
    if Defend_Status:
        #print('DEFENDING')
        defense_X_test, defense_y_test,defense_conf, defense_locations = check_dataset(X_test,y_test,tolerance=defense_tolerance)
        experiment_data['defend_locs'] = defense_locations
        experiment_data['defend_conf'] = defense_conf
        experiment_data['defend_labels'] = defense_y_test
        X_test = defense_X_test
        y_test = defense_y_test
    naive_weights=(NaiveAnalyzer(X_test['text'])) 
    experiment_data['naive_sentiments'] = [1 if w > 0 else -1 for w in naive_weights]
    experiment_data['naive_sentiments_conf'] = [abs(w) for w in naive_weights]
    
    log_label, log_conf = logistic_analyze(X_train['text'],y_train,X_test['text'], random_state= random_state_int)
    experiment_data['logistic_regression']=(log_label)
    experiment_data['logistic_regression_conf'] = log_conf

    dl_label, dl_conf = DeepLearningAnalyzer(X_test['text'])
    experiment_data['deep_learning']=(dl_label)
    experiment_data['deep_learning_conf']=(dl_conf)

    trans_label, trans_conf = TransformerAnalyzer(X_test['text'])
    experiment_data['transformer']=(trans_label)
    experiment_data['transformer_conf']=(trans_conf)
    #print(experiment_data)
    return experiment_data
if __name__ == '__main__':
    experiment(Attack_Status=True, Defend_Status= True)
    