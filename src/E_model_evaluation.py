
import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report
import logging
import mlflow

import dagshub
dagshub.init(repo_owner='priyanshu24003', repo_name='DataV_MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/priyanshu24003/DataV_MLFlow.mlflow")


# Ensure the "logs" directory exists
log_dir = 'data/logs'

os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('E_model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'E_model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


with open('data/logs/E_model_evaluation.log') as f:
    lines = f.readlines()
    init_log_length = len(lines)


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data (usually test data) from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        # y_pred_proba = clf.predict_proba(X_test)[:, 1]

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

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
        logger.debug('\n')

    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        clf = load_model('./data/models/model.pkl')
        test_data = load_data('./data/processed/test_final.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
                
        metrics = evaluate_model(clf, X_test, y_test)

        ## logging metrics
        mlflow.log_metric('accuracy',metrics['accuracy'])
        ## logging metrics

        ##logging comaparion , child models we had saved earliear 
        # for key in basic_Models.keys():
        #     with mlflow.start_run(nested=True) as child:
        
        save_metrics(metrics, './data/reports/metrics.json')

        try:
            logger.debug('nested logging started')
            ## logging accuracies of basic models for comparisions
            folder = "./data/models/basic_M"   # ← change this

            file_paths = [
                os.path.join(folder, filename)
                for filename in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, filename))
            ]

            for i,ms in enumerate(file_paths):
                sub_model = load_model(ms)
                metricsss = evaluate_model(sub_model, X_test,y_test)
                mlflow.log_metric(f'accuracy {ms[-6:-4]}', metricsss['accuracy'])

                #logging inside child runs
                with mlflow.start_run(nested=True) as child:
                    
                    mlflow.sklearn.log_model(sub_model, name=ms[-6:-4])
                    mlflow.log_metric('accuracy', metricsss['accuracy'])
                    mlflow.set_tag('model_name',f'{ms[-6:-4]}')

                logger.debug('nested logging Finished')

        except Exception as e:
            logger.error('Error while doing nested logging for LR, RF comparision')

        logger.debug('report generated successfully')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

    with open("data/logs/E_model_evaluation.log") as f2:
        liness = f2.readlines()
        
        with open('data/current_exp.log', 'a') as f3:
            f3.writelines(liness[init_log_length:])




if __name__ == '__main__':
    main()
