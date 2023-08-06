import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os
from src.utility import Utils
from src.logs import logger 
import argparse
import traceback



class DataIngestion:

    def __init__(self):
        self.utils= Utils()
        


    def save_csv_to_directory(self, data_frame, file_path):

        try:
            data_frame.to_csv(file_path, index=False)
            logger.info(f"file  saved to '{file_path}'.")

        except:
            logger.error(traceback.print_exc())

    def _init_data_ingetion(self, config_path):

        try:
            config = self.utils.read_params(config_path)
            df= self.utils.load_data(config['data_source']['data_url'])
            random_state= config['data_source']['random_state']

            df_train, df_val, df_test = np.split(df.sample(frac=1, random_state= random_state),
                                        [int(.93*len(df)), int(.99*len(df))])
            logger.info(df_train.shape)
            logger.info(df_test.shape)
            logger.info(df_val.shape)

            self.utils.create_directory(config['data_source']['directory_path'])

            raw_data_path= config['load_data']['raw_dataset_path']
            train_path= config['load_data']['train_path']
            test_path= config['load_data']['test_path']
            val_path= config['load_data']['val_path']

            self.save_csv_to_directory(df, raw_data_path)
            self.save_csv_to_directory(df_train, train_path)
            self.save_csv_to_directory(df_test, test_path)
            self.save_csv_to_directory(df_val, val_path)
        
        except:
            logger.error(traceback.print_exc())


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    ingestion= DataIngestion()
    ingestion._init_data_ingetion(config_path=parsed_args.config)