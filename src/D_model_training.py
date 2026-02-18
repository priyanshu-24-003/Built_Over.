import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from sklearn import svm
from sklearn.model_selection import cross_val_score
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import dagshub
dagshub.init(repo_owner='priyanshu24003', repo_name='DataV_MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/priyanshu24003/DataV_MLFlow.mlflow")



# Ensure the "logs" directory exists
log_dir = 'data/logs'

os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('D_Model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'D_Model_Training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


with open('data/logs/D_Model_Training.log') as f:
    lines = f.readlines()
    init_log_length = len(lines)


def basic_model_training(X_train, Y_train):
    """
    Docstring for basic_model_training
    
    :param X_train: Features
    :param Y_train: Labels

    Trains basic models for later comparision of main model
    """
    basic_Models = {"RF":RandomForestClassifier(), "LR":LogisticRegression()}

    try:
        logger.debug('training random forest and logistic regression for comparision')
        
        for key in basic_Models.keys():
                basic_Models[key].fit(X_train, Y_train)
                model_save_path = f'data/models/basic_M/{key}.pkl'
                save_model(basic_Models[key], model_save_path)
                mlflow.sklearn.log_model(basic_Models[key], name=key)

        logger.debug('Basic model Training has been compleated')
        
    except Exception as e:
        logger.error('error while training basic models')
        raise
        
    return basic_Models
    

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise



def train_model(X_train: np.ndarray, y_train: np.ndarray,) -> svm.SVC:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:                        
        gammas = np.linspace(0.05, 5, 100)

        scores = []
        best_score = -np.inf

        for g in gammas:
            SVM = svm.SVC(gamma = g)

            scores.append(cross_val_score(SVM, X_train, y_train, cv = 5).mean())
            
            if scores[-1] > best_score:
                best_score = scores[-1]
                best_gamma = g
                
        best_gamma, best_score

        SVM = svm.SVC(gamma = best_gamma, kernel='rbf', C=1.0, probability=True)

        SVM.fit(X_train, y_train)
        
        logger.debug(f'model has been trained with score on training data :{SVM.score(X_train, y_train)} ')

        basic_model_training(X_train, y_train)

        return SVM

    
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise



def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
        logger.debug('\n')

    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model in the pickle file: %s', e)
        raise



def main():
    try:

        train_data = load_data('./data/processed/train_final.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train,)
        
        #loggin the model in MLFlow
        mlflow.sklearn.log_model(clf, 'SupportVectorClassifier')
        #loggin the model in MLFlow


        model_save_path = 'data/models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

    with open("data/logs/D_Model_Training.log") as f2:
        liness = f2.readlines()
        
        with open('data/current_exp.log', 'a') as f3:
            f3.writelines(liness[init_log_length:])

  

if __name__ == '__main__':
    main()