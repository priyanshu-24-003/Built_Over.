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
                Savy(model_save_path, 'model', logger, __file__, basic_Models[key],).save_it()

        logger.debug('Basic model Training has been compleated')
        
    except Exception as e:
        logger.error('error while training basic models')
        raise
        
    return basic_Models

def check_models(X_train_small, y_train_small,x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    try:
        logger.debug('multimodel training started with small training data')

        m1 = LogisticRegression()
        m1.fit(X_train_small, y_train_small)
        s1 = m1.score(x_test, y_test)

        
        m2 = DecisionTreeClassifier()
        m2.fit(X_train_small, y_train_small)
        s2 = m2.score(x_test,y_test)


        m3 = KNeighborsClassifier()
        m3.fit(X_train_small, y_train_small)
        s3 = m3.score(x_test, y_test)

        score2model = {s1:m1, s2:m2, s3:m3}
        best = max(score2model.keys())
        logger.debug(f'the best out of 3 models has been trained and returned {score2model[best]} with mini testing score {best}')
        return score2model[best]

    except Exception as e:
        logger.error("error line 104, multi model training , ")
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

def main():
    try:
        params = load_params('params.yaml')['model_building']

        train_data = Load('./data/processed/train.csv', 'df', logger, __file__).load_it()

        if params['model_types'] == "main":
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values

            clf = train_model(X_train, y_train,)
        elif params['model_types'] == 'multi':
            test_data = Load('./data/processed/test.csv', 'df', logger, __file__).load_it()
            x_test = test_data.iloc[:10, :-1].values
            y_test = test_data.iloc[:10, -1].values
            clf = check_models(train_data.iloc[:100, :-1], train_data.iloc[:100, -1], x_test, y_test)
            

        model_save_path = 'data/models/model.pkl'
        Savy(model_save_path, 'model', logger, __file__, clf,).save_it()

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

  

if __name__ == '__main__':
    main()