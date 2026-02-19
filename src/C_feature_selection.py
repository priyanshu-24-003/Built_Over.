from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
import logging
import yaml
import numpy as np
import mlflow

#importing helper module
from helper.Loader import Load
from helper.Saver import Savy


# import dagshub
# dagshub.init(repo_owner='priyanshu24003', repo_name='DataV_MLFlow', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/priyanshu24003/DataV_MLFlow.mlflow")



# Ensure the "logs" directory exists
log_dir = 'data/logs'

os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('C_Feature_selection')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'C_Feature_selection.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


with open('data/logs/C_Feature_selection.log') as f:
    lines = f.readlines()
    init_log_length = len(lines)



def check_column_scores(cols, df):
    """
    Trains and evaluates Logistic Regression via crossvalidation on the columns
    of the dataset with select indeces
    
    Parameters
    ----------
    cols : list of strings, columns on which to be trained
    
    Return
    ----------
    float : average of 5 cross validation scores
    """
    
    #Logistic Regression Model
    LR = LogisticRegression(max_iter = 5000)
    
    df2 = df.copy()
    y = df['Species']
    X = df2.drop(["Species"], axis = 1)

    return cross_val_score(LR, X[cols], y, cv = 5).mean()

def FeatureSelection(df):
    try:
        logger.debug('feature selection started ')
        quals = ["Island", "Sex"]

        quants = ["Culmen Length (mm)", "Culmen Depth (mm)",
                "Flipper Length (mm)", "Body Mass (g)"]

        combos = [[qual]+[quant1]+[quant2] for qual in quals for quant1 in quants for quant2 in quants if quant1 != quant2]

        cv_scores = []
        best_cv_score = -np.inf

        for combo in combos:
            score = check_column_scores(combo, df)
            cv_scores.append(score)
            
            if cv_scores[-1] > best_cv_score:
                best_cv_score = cv_scores[-1]
                best_combo = combo

        logger.debug("Best Feature selected which produces CV score: " + str(best_cv_score))

        return best_combo
    except:
        logger.error('error occured while selecting best featueres')
        raise
    pass


def main():
    try:

        train_data = Load('./data/interim/train.csv', 'df', logger, __file__).load_it()
        test_data = Load('./data/interim/test.csv', 'df', logger, __file__).load_it()

        BestFeatures = FeatureSelection(train_data) + ['Species']

        train_df = train_data[BestFeatures]
        test_df = test_data[BestFeatures]

        #Saving data
        save_path = os.path.join("./data", "processed")
        Savy(save_path, 'df', logger, __file__, train_df, test_df).save_it()

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")
 
    with open("data/logs/C_Feature_selection.log") as f2:
        liness = f2.readlines()
        
        with open('data/current_exp.log', 'a') as f3:
            f3.writelines(liness[init_log_length:])

 

if __name__ == '__main__':
    main()