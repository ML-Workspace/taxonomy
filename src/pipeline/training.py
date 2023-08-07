import pandas as pd
import yaml
import os
from src.utility import Utils
from src.logs import logger 
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
from src.component.data_processing import TextPreprocessor
import traceback
import sys
import argparse


class TrainingPipeline:

    def __init__(self):
        self.model_name =  'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.accuracy = evaluate.load("accuracy")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.processor= TextPreprocessor()
        self.utils= Utils()

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def generate_label_id(self, file_path):
        id2label = {}
        label2id = {}
        df = pd.read_csv(file_path)

        for i, item in enumerate(df['labels'].unique()):
            id2label[i] = item
            label2id[item] = i

        print('Successfully generated the label ids!')
        return label2id, id2label

    def read_training_data(self, data_file_path, train_file_path, val_file_path):
        df_train = pd.read_csv(train_file_path)
        df_val = pd.read_csv(val_file_path)


        df_train['text']= df_train['text'].apply(lambda x : self.processor.text_preprocessing(x))
        df_train= df_train[['text', 'labels']]

        df_val['text']= df_val['text'].apply(lambda x : self.processor.text_preprocessing(x))
        df_val= df_val[['text', 'labels']]
        

        label2id, id2label = self.generate_label_id(data_file_path)

        df_train['labels']= df_train['labels'].map(label2id)
        df_val['labels']= df_val['labels'].map(label2id)

        df_train = Dataset.from_pandas(df_train)
        df_val = Dataset.from_pandas(df_val)



        train_tokenized = df_train.map(self.preprocess_function, batched=True)
        val_tokenized = df_val.map(self.preprocess_function, batched=True)
        print(train_tokenized['text'])
        print('\n\n')
        print(train_tokenized['labels'])

        print('Successfully tokenized the data for training !!!!!!')

        return train_tokenized, val_tokenized

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)

    def training(self, config_path):
        try:
            config = self.utils.read_params(config_path)

            data_file_path= config['data_source']['data_url']
            train_file_path= config['load_data']['train_path']
            val_file_path= config['load_data']['val_path']
            num_labels= config['estimators']['num_labels']
            model_output_directory= config['estimators']['model_dir']
            learning_rate= config['estimators']['params']['learning_rate']
            train_batch_size= config['estimators']['params']['train_batch_size']
            eval_batch_size= config['estimators']['params']['eval_batch_size']
            epochs= config['estimators']['params']['epochs']
            weight_decay= config['estimators']['params']['weight_decay']
            evaluation_strategy= config['estimators']['params']['evaluation_strategy']
            save_strategy= config['estimators']['params']['save_strategy']


            label2id, id2label = self.generate_label_id(data_file_path)
            train_tokenized, val_tokenized = self.read_training_data(data_file_path,train_file_path, val_file_path)

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels= num_labels, id2label=id2label, label2id=label2id
            )

            training_args = TrainingArguments(
                output_dir=model_output_directory,
                learning_rate= learning_rate,
                per_device_train_batch_size=  train_batch_size,
                per_device_eval_batch_size= eval_batch_size,
                num_train_epochs= epochs,
                weight_decay= weight_decay,
                evaluation_strategy= evaluation_strategy,
                save_strategy= save_strategy,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )

            trainer.train()
            print('Successfully started the training !!!!!!')

        

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(tb)


if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    pipeline = TrainingPipeline()
    pipeline.training(config_path=parsed_args.config)
