import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from pathlib import Path 
sys.path.insert(0, 'D:\DiamondPricePrediction\src')
from src.logger import *
from src.exception import *
from src.utils import *

# DATA TRANSFORMATION CONFIG

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

# data TRANSFORMATION class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    def get_data_transformation_object(self):
        try:
            logging.info("data transformation initiated")
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']
            # defining the rank
            cut_categories=["Fair","Good","Very Good","Premium","Ideal"]
            color_categories=["D","E","F","G","H","I","J"]
            clarity_categories=["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
            logging.info("PIPELINING INITIATED")

            ## NUMERICAL PIPELINE
            num_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='median')),  ## Missing values
            ('scaler',StandardScaler())  ## feature scaling
            ]
            )
            ## categorical pipeline
            cat_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),  ## Missing values
            ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])) ,## categorical features to numerical features
            ('scaler',StandardScaler()) ]
            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
    
            ])
            return preprocessor
            logging.info('Pipeline completed')
        except Exception as e:
            logging.info("ERROR IN DATA TRANFORMATION")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            ## features into independent and dependent features

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]


            ## apply the transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            save_objects(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)