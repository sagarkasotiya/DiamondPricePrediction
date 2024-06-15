import os
import sys
import pandas as pd
sys.path.insert(0, 'D:\DiamondPricePrediction\src')
from src.logger import *
from src.exception import *
#from logger import logging
#from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# initialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

# create data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info('data ingestion method starts')

        try:
            logging.info('1')
            df=pd.read_csv(os.path.join('notebooks\data','gemstone.csv'))
            #df=pd.read_csv('D:\DiamondPricePrediction\notebooks\data\gemstone.csv')
            logging.info('2')
            logging.info('dataset read as pandas dataframe')
            #os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('TRAIN TEST SPLIT')
            train_set,test_set=train_test_split(df,test_size=0.30)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

            


        except Exception as e:
            logging.info('ERROR HAS OCUURRED IN DATA INGESTION')
    