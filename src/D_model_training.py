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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


#importing helper module from parent
from helper.Loader import Load
from helper.Saver import Savy
from helper.Logster import Logy


LD = Logy('D_Model_Training.log','data')
logger = LD.get_this_logy()

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def check_models(Xtrain, Ytrain, x_val, y_val):

    try:
        logger.debug('multimodel training started with training data')

        models = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier()]
        score2model = {}
        for m in models:
            m.fit(Xtrain, Ytrain)
            score2model[m.score(x_val, y_val)] = m

        best = max(score2model.keys())
        logger.debug(f'the best out of 3 models has been trained and returned {score2model[best]} with score on validation dataset {best}')
        return score2model[best]

    except Exception as e:
        logger.error("error line 104, multi model training , ")
        raise 

def train_model(X_train: np.ndarray, y_train: np.ndarray,) -> svm.SVC:
    """
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained SupportVectorClassifier
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

        return SVM

    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')['model_building']

        train_data = Load('./data/processed/train.csv', 'df', logger, __file__).load_it()
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        if params['model_types'] == "tuned":     

            clf = train_model(X_train, y_train,)


        elif params['model_types'] == 'baseline':

            test_data = Load('./data/processed/test.csv', 'df', logger, __file__).load_it()
            x_val = test_data.iloc[:10, :-1].values
            y_val = test_data.iloc[:10, -1].values
            clf = check_models(X_train, y_train, x_val, y_val)
            

        model_save_path = 'data/models/model.pkl'
        Savy(model_save_path, 'model', logger, __file__, clf,).save_it()

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

  

if __name__ == '__main__':
    main()