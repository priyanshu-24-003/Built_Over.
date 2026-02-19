import json
import pandas as pd
import pickle 
import logging
import os

class Save():

    def __init__(self,file_path:str, file_type:str, logger:logging.getLogger, component):
        self.path = file_path
        self.typ = file_type
        self.log = logger
        self.component = component.split('\\')[-1]

        os.makedirs(os.path.dirname(self.path), exist_ok=True)


class Savy(Save):
    
    def __init__(self, file_path, file_type, logger, component, *data):
        super().__init__(file_path, file_type, logger, component, *data)
        self.data = data
    
    def save_it(self,):
        try:
            if self.typ == 'df':
                pass
        
        except:
            pass
    
    def save_model(self, model):

        with open(self.path, 'wb') as file:
            pickle.dump(model, file)
        self.log.debug('Model saved to %s', self.path)
        self.log.debug('\n')
    
    def save_report_metrics(self, metrics):

        with open(self.path, 'w') as file:
            json.dump(metrics, file, indent=4)
        self.log.debug('Metrics saved to %s', self.path)
        self.log.debug('\n')

    def save_data(self, train_data, test_data):
        train_data.to_csv(os.path.join(self.path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(self.path, "test.csv"), index=False)
        self.log.debug('Train and test data saved to %s', self.path)
        self.log.debug('\n')
