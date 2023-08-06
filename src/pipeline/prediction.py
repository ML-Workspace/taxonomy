from transformers import pipeline
import os
import argparse
from src.utility import Utils

class Predictions:

    def __init__(self, config_path):
        self.utils= Utils()
        self.config= self.utils.read_params(config_path)
        self.model_path = os.path.join(self.config['estimators']['model_dir'], 'taxonomy_model')
        self.classifier = pipeline("sentiment-analysis", model=self.model_path)

    def predict_sentiment(self, text):
        prediction = self.classifier(text)[0]['label']  
        return prediction

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    predict= Predictions(config_path= parsed_args.config)
    text= '''New website is 'Airbnb of yoga' reflecting the rise and rise of yoga retreats. 
    A new online booking platform is set to try and tap into the wellness tourism market by 
    positioning itself as the ‘Airbnb of Yoga’.'''
    value= predict.predict_sentiment(text)
    print(value)

    