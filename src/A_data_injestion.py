import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
from sklearn.model_selection import train_test_split
import mlflow

#importing helper module from parent
from helper.Loader import Load
from helper.Saver import Savy
from helper.Logster import Logy

#Remote server tracking using dagshub
# import dagshub
# dagshub.init(repo_owner='priyanshu24003', repo_name='DataV_MLFlow', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/priyanshu24003/DataV_MLFlow.mlflow")

LA = Logy('data_ingestion.log','data')
logger = LA.get_this_logy()


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

def main():    
    data = Load("data/src/palmer_penguins.csv", 'df', logger=logger, component=__file__)
    df = data.load_it()
    
    df = basic_processing(df)

    test_siz = 0.2

    XY, xy = train_test_spliter(df, test_siz, 42)
    
    #logging param test_size
    mlflow.log_param('test_size', test_siz)

    Savy('./data/raw', 'df', logger, __file__, XY, xy).save_it()

  

if __name__ == '__main__':
    main()
