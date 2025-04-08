import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
     def __init__(self):
          pass
     
     def predict(self, features):
          try:
               preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
               model_path = os.path.join('artifacts', 'model.pkl')
               
               preprocessor = load_object(preprocessor_path)
               model = load_object(model_path)
               
               data_scaled = preprocessor.transform(features)
               
               pred = model.predict(data_scaled)
               return pred
               
          except Exception as e:
               logging.info("Exception occured in prediction")
               raise CustomException(e, sys)
          
class CustomData:
     
     def __init__(self,
                  Amount : float,
                  Campaign_Name : str,
                  Sub_Id : str,
                  Partner : str):
          
          self.Amount = Amount
          self.Campaign_Name = Campaign_Name
          self.Sub_Id = Sub_Id
          self.Partner = Partner 
          
          
     def get_data_as_dataframe(self):
          try:
               custom_data_input_dict = {
                    'Amount': [self.Amount],
                    'Campaign_Name': [self.Campaign_Name],
                    'Sub_Id': [self.Sub_Id],
                    'Partner' : [self.Partner],
               }
               
               df = pd.DataFrame(custom_data_input_dict)
               logging.info("Data Frame Gathered")
               return df
          
          except Exception as e:
               logging.info("Exception Occured in prediction pipeline")
               raise CustomException(e, sys)
               
          
          
               