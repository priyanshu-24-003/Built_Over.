from sklearn import preprocessing
import os
import logging
import pandas as pd
import mlflow

#importing helper module
from helper.Loader import Load
from helper.Saver import Savy
from helper.Logster import Logy

LB = Logy('B_preparation.log','data')
logger = LB.get_this_logy()


def prep_penguins_data(data, trORte):
    """
    Prepares penguins data frame for machine learning models.
    
    Parameters
    ----------
    data : pandas.DataFrame to be prepared
    
    Return
    ----------
    (X, y) : pandas.DataFrame without Species column, pandas.Series of Species column
    """
    
    df = data.copy()
    
    try:
        cats = ["Sex", "Island", "Species",]
        for c in cats:
            le = preprocessing.LabelEncoder()
            
            df[c] = le.fit_transform(df[c])
            
        logger.debug(f'successfully encoded {cats} the catagorical features of {trORte}')

        return df
        
    except Exception as e:
        print(df.sample(2))
        logger.error(f'error in prep_penguins_data {e}')
        raise 


def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:

        train_data = Load('./data/raw/train.csv', 'df', logger, __file__).load_it()
        test_data = Load('./data/raw/test.csv', 'df', logger, __file__).load_it()


        train_processed_data = prep_penguins_data(train_data, 'train data', )
        test_processed_data = prep_penguins_data(test_data,'test data')

        data_path = os.path.join("./data", "interim")
        Savy(data_path, 'df', logger, __file__, train_processed_data, test_processed_data).save_it()

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
