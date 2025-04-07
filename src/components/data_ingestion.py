from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
     
     train_data_path:str = os.path.join('artifacts', 'train.csv')
     test_data_path:str = os.path.join('artifacts', 'test.csv')
     raw_data_path:str = os.path.join('artifacts', 'raw.csv')
     
# create a class for Data Ingestion

class DataIngestion:
     def __init__(self):
          self.ingestion_config = DataIngestionconfig()
          
     def initiate_data_ingestion(self):
          logging.info('Data Ingestion methods Starts')
          try:
               df = pd.read_csv('notebooks/data/Camp.csv')
               logging.info("Dataset read as pandas DataFrame")
               
               os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
               df.to_csv(self.ingestion_config.raw_data_path, index=False)
               logging.info("Train test split")
               train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
               
               train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
               test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
               logging.info("Data Ingestion Completed")
               
               return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
               )
          
          except Exception as e:
               logging.info('Exception occured at Data Ingestion stage')
               raise CustomException(e, sys)
          
               
          
# Run Data
if __name__ == "__main__":
     obj = DataIngestion()
     train_data, test_data = obj.initiate_data_ingestion()