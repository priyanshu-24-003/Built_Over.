
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


# import dagshub
# dagshub.init(repo_owner='priyanshu24003', repo_name='DataV_MLFlow', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/priyanshu24003/DataV_MLFlow.mlflow")


# Ensure the "logs" directory exists


LE = Logy('E_model_evaluation.log','data')
logger = LE.get_this_logy()


with open('data/logs/E_model_evaluation.log') as f:
    lines = f.readlines()
    init_log_length = len(lines)


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


def main():
    try:
        clf = Load('./data/models/model.pkl', 'model', logger, __file__).load_it()
        test_data = Load('./data/processed/test.csv', 'df', logger, __file__).load_it()
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
                
        metrics = evaluate_model(clf, X_test, y_test)

        ## logging metrics
        mlflow.log_metric('accuracy',metrics['accuracy'])
        ## logging metrics

        ##logging comaparion , child models we had saved earliear 
        # for key in basic_Models.keys():
        #     with mlflow.start_run(nested=True) as child:
        
        print(metrics)
        save_metric = Savy('./data/reports/metrics.json', 'report', logger, __file__, metrics,)
        save_metric.save_it()

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
                sub_model = Load(ms,'model', logger, __file__).load_it()
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
