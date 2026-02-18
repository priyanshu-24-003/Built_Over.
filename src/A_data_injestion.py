import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
from sklearn.model_selection import train_test_split
import mlflow

#importing extra
from helper.Loader import Load

#Remote server tracking using dagshub
# import dagshub
# dagshub.init(repo_owner='priyanshu24003', repo_name='DataV_MLFlow', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/priyanshu24003/DataV_MLFlow.mlflow")



# creating a log file for later debbuging.

log_dir = 'data/logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


with open('data/logs/data_ingestion.log') as f:
    lines = f.readlines()
    init_log_length = len(lines)


#data loader
# def load_data(data_url: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(data_url)
#         logger.debug('Data loaded from %s', data_url)
#         return df
    
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise


def train_test_spliter(df, test_size, random_state):
    try:
        XY, xy = train_test_split(df, test_size=test_size, random_state=random_state)
        logger.debug('split the data into training and testing')
        return (XY, xy)
    except:
        logger.error('error in train_test_spliter function')
        
def basic_processing(df):
    #Look at just the Species, Island, Culmen Length, Culmen Depth, Flipper Length, Body Mass, and Sex Columns
    df = df.drop(["studyName", "Sample Number", "Region", "Stage", "Individual ID", "Clutch Completion",
                "Date Egg", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Comments"], axis = 1)

    #Shorten the species names to Adelie, Gentoo, and Chinstrap
    df["Species"] = df["Species"].str.split().str.get(0)

    #Drop the one row where Sex was recorded as "."
    df = df[df["Sex"] != "."]

    #Drop rows that include NaN 
    df = df.dropna(subset = ["Sex"])

    return df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = data_path
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
        logger.debug('\n')
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():    
    # df = load_data('data/src/palmer_penguins.csv')
    
    
    data = Load("data/src/palmer_penguins.csv", 'df', logger=logger)
    df = data.load_it()
        
    
    df = basic_processing(df)

    test_siz = 0.2

    XY, xy = train_test_spliter(df, test_siz, 42)
    
    #logging param test_size
    mlflow.log_param('test_size', test_siz)

    save_data(XY, xy, './data/raw')

    with open("data/logs/data_ingestion.log") as f2:
        liness = f2.readlines()
        
        with open('data/current_exp.log', 'a') as f3:
            f3.writelines(liness[init_log_length:])


if __name__ == '__main__':
    main()
