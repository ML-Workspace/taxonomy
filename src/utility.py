import os
import pandas as pd
import yaml

class Utils:

    def __init__(self):
        pass

    def read_params(self, config_path):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    def create_directory(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")

    def load_data(self, url):
        data = pd.read_csv(url, sep=',')
        return data

