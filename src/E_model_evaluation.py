
import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report
import logging
import mlflow

#importing helper module from parent
from helper.Loader import Load
from helper.Saver import Savy
from helper.Logster import Logy

## Local trac# Local tracking server setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Local tracking server setup


LE = Logy('E_model_evaluation.log','data')
logger = LE.get_this_logy()



def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics_dict = {
            'accuracy': accuracy,
            'Multiclass report': report ,
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def main():
    """
    Experiment Tracking using mlfow in this Model_Evaluation component. 
    """
    try:
        mlflow.set_experiment('Experiment_1_classification_model')
        with mlflow.start_run() as run:

            clf = Load('./data/models/model.pkl', 'model', logger, __file__).load_it()
            test_data = Load('./data/processed/test.csv', 'df', logger, __file__).load_it()
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
                    
            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metric = Savy('./data/reports/metrics.json', 'report', logger, __file__, metrics,)
            save_metric.save_it()

            # Logging data neccessory for experiment tracking

            #log metrics
            mlflow.log_metric('accuracy',metrics['accuracy'], )

    

            #log the training and testing data
            data_set = mlflow.data.from_pandas(test_data, 'test_data')
            mlflow.log_input(data_set, 'test_data')

            train_data = Load('./data/processed/train.csv', 'df', logger, __file__).load_it()
            train_data_set = mlflow.data.from_pandas(train_data, 'train_data')
            mlflow.log_input(train_data_set, 'train_data')
            #log the training and testing data

            #logging the paramsfile
            mlflow.log_artifact('./params.yaml')

            #log the model 
            mlflow.sklearn.log_model(clf, 'Model')

            #log the reports
            mlflow.log_artifact('./data/reports/metrics.json')

            logger.debug('report generated successfully')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")




if __name__ == '__main__':
    main()
