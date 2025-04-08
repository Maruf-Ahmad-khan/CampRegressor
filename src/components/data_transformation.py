import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os


@dataclass
class DataTransformationConfig:
     preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
     
     
class DataTransformation:
     def __init__(self):
          self.data_transformation_config = DataTransformationConfig()
          
          
     def get_data_transformation_object(self):
          try:
               logging.info("Data Transformation initiated")
               
               categorical_cols = ['Campaign_Name', 'Sub_Id', 'Partner']
               numerical_cols = ['Amount']
               
               
               # Define the custom ranking for each ordinal variable
               Campaign_categories = ['Pepperfry', 'Shopclues IN', 'Boat', 'Fernsnpetals', 'Puma', 'Firstcry', 'GoDaddy', 'Croma Retail', 'Myntra', 'Ajio', 'Flipkart']
               Sub_Id_categories = ['web', 'store_page', 'telegram', 'app']
               Partner_categories = ['Cuelinks', 'Admitad']
               
               logging.info("Pipeline Initiated")
               
               
               # Numerical Pipeline
               num_pipeline = Pipeline(
                         steps=[
                         ('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())   
                         
                         ]    
               ) 

               # Categorical Pipeline

               cat_pipeline = Pipeline(
                         steps=[
                              ('imputer', SimpleImputer(strategy='most_frequent')),
                              ('ordinal_encoder', OrdinalEncoder(categories=[
                              Campaign_categories,Sub_Id_categories, Partner_categories],
                              handle_unknown='use_encoded_value',unknown_value=-1)),
                              ('scaler', StandardScaler())
                         ]
                    
               )

               preprocessor = ColumnTransformer([
               ('num_pipeline', num_pipeline, numerical_cols),
               ('cat_pipline', cat_pipeline, categorical_cols)
               ])
               
               return preprocessor
               logging.info("Pipeline Completed")

          
          except Exception as e:
               logging.info("Error in Data Transformation")
               raise CustomException(e, sys)
          
          
          
     def initiate_data_transformation(self, train_path, test_path):
          try:
               
               train_df = pd.read_csv(train_path)
               test_df = pd.read_csv(test_path)
               
               
               logging.info("Read train and test data completed")
               logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
               logging.info(f"Test DataFrame Head : \n{test_df.head().to_string()}")
               logging.info("Obtaining preprocessing object")
               
               preprocessing_obj = self.get_data_transformation_object()
               
               target_column_name = 'Payout'
               drop_columns = [target_column_name, 'Id', 'Status', 'Date',
                               'Currency', 'Inventory', 'Email_Id']
               input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
               target_feature_train_df = train_df[target_column_name]
               
               
               input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
               target_feature_test_df = test_df[target_column_name]
               
               input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
               input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
               
               logging.info("Applying preprocessing object on training and testing datasets.")
               
               
               train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
               test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
               
               save_object(
                    
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
               )
               logging.info("Preprocessor pickle file saved")
               
               return train_arr,test_arr, self.data_transformation_config.preprocessor_obj_file_path
               
          
          except Exception as e:
               logging.info("Exception occured in the initiate_data_transformation")
               raise CustomException(e, sys)
          