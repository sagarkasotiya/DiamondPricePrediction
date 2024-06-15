import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    try:
        # Data Ingestion
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print(train_data_path, test_data_path)
        
        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        print(train_arr, test_arr, preprocessor_obj_file_path)
        
        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
    
    except Exception as e:
        logging.error('Exception occurred during the pipeline execution', exc_info=True)
        raise CustomException(e, sys)
