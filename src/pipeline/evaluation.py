from transformers import pipeline
from src.utility import Utils
import os
import argparse
import pandas as pd 
from sklearn.metrics import f1_score
from src.logs import logger

class Evaluate:

    def __init__(self, config_path):
        self.utils= Utils()
        self.config= self.utils.read_params(config_path)
        self.model_path = os.path.join(self.config['estimators']['model_dir'], 'taxonomy_model')
        self.classifier = pipeline("sentiment-analysis", model=self.model_path)


    def generate_label_id(self, file_path):
        id2label = {}
        label2id = {}
        df = pd.read_csv(file_path)

        for i, item in enumerate(df['labels'].unique()):
            id2label[i] = item
            label2id[item] = i

        print('Successfully generated the label ids!')
        return label2id, id2label


    def model_evaluation(self):

        predicted_labels= []

        df_test= self.utils.load_data(self.config['load_data']['test_path'])
        file_path= self.config['load_data']['raw_dataset_path']

        _, id2label= self.generate_label_id(file_path)

        model= os.path.join(self.config['estimators']['model_dir'], 'taxonomy_model')

        for item in df_test.index:
            text = df_test['text'][item]
            predicted_labels.append(self.classifier(text)[0]['label'])
            self.classifier.save_pretrained(self.config['estimators']['pipeline_dir'])

        actual_labels= list(df_test['labels'])


        f1_scores = f1_score(actual_labels, predicted_labels, average= None, labels=actual_labels)
        f1_score_dict = {class_label: f1 for class_label, f1 in zip(actual_labels, f1_scores)}

        return f1_score_dict

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    evaluation = Evaluate(config_path= parsed_args.config)
    score= evaluation.model_evaluation()
    