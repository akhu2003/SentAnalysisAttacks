from transformers import pipeline
import pandas as pd

def check_dataset(dataset,y_test, tolerance):
    ''''
    returns truncated dataset, confidence for whole dataset
    '''
    dataset = dataset['text'].to_list()
    detector = pipeline("text-classification",
                    model="roberta-base-openai-detector")
    modified_dataset = []
    modified_y_test = []
    confidence = []
    real_locs = []
    for i in range(len(dataset)):
        result = detector(dataset[i])
        l = result[0]['label']
        c = result[0]['score']
        if c > tolerance and l == 'Fake':
            real_locs.append(False)
            continue
        modified_dataset.append(dataset[i])
        confidence.append(c)
        modified_y_test.append(y_test[i])
        real_locs.append(True)
    return pd.DataFrame({'text':modified_dataset}),modified_y_test, confidence, real_locs