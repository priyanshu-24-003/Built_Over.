import json
import pandas as pd
import pickle 
import logging
import os
from pathlib import Path


def get_filename(path_str: str) -> str:
    return Path(path_str).name


class Save():

    """
        A class to Save Different types of Artifacts in the Pipeline.
    """

    def __init__(self,file_path:str, file_type:str, logger:logging.getLogger, component):
        """
        file_type: ['df', 'model', 'report']
        file_path: file_type gets stored here (could be a dir, or simple a file_path)
        logger: Configured logging entity that saves logs for the component.
        component: Component of the Pipeline (A, B, C, D, E. (__File__) at the time of execution)

        constructor creates the directory if it doesn't exist ; ignores if it does.
        """

        self.path = file_path
        self.typ = file_type
        self.log = logger
        self.component = get_filename(component)

        if '.' not in self.path[2:]:
            os.makedirs(self.path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

class Savy(Save):
    
    def __init__(self, file_path, file_type, logger, component, *data):
        """
        data : tuple of artifacts of same file_type to be saved.
        """

        super().__init__(file_path, file_type, logger, component,)
        self.data = data
    
    def save_it(self,):
        """
        A compiler method that saves the artifacts according to their types
        """

        try:
            if self.typ == 'df':
                self.save_data(self.data[0], self.data[1])
            elif self.typ == 'model':
                self.save_model(self.data[0])
            elif self.typ == 'report':
                self.save_report_metrics(self.data[0])

        except Exception as e:
            self.log.error(f'Unexpected error occurred while Saving the {self.typ} in {self.component}, {e}')
            raise

    
    def save_model(self, model):
        """
        Method to save model using pickle.dump .
        """

        with open(self.path, 'wb') as file:
            pickle.dump(model, file)
        self.log.debug('Model saved to %s', f' {self.path} inside {self.component}')
        self.log.debug('\n')
    
    def save_report_metrics(self, metrics):
        """
        Method to save the report using json.dump .
        """

        with open(self.path, 'w') as file:
            json.dump(metrics, file, indent=4)
        self.log.debug('Metrics saved to %s', f' {self.path} inside {self.component}')
        self.log.debug('\n')

    def save_data(self, train_data, test_data):
        """
        Method to save the data using pandas.DataFrame.to_csv .
        """

        train_data.to_csv(os.path.join(self.path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(self.path, "test.csv"), index=False)
        self.log.debug('Train and test data saved to %s', f' {self.path} inside {self.component}')
        self.log.debug('\n')
