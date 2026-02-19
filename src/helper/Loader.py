import pandas as pd
import pickle 
import logging

class Load():

    def __init__(self,file_path:str, file_type:str, logger:logging.getLogger, component):
        self.path = file_path
        self.typ = file_type
        self.log = logger
        self.component = component.split('\\')[-1]

    def load_it(self,):
        self.log.debug(f'started Data_loading inside component {self.component}')
        try:
            if self.typ == 'model':

                with open(self.path, 'rb') as file:
                    self.log.debug(f'model Loaded successfully inside component {self.component}')
                    return pickle.load(file)
                
            elif self.typ == 'df':
                self.log.debug(f'Data Loaded successfully inside component {self.component}')
                return pd.read_csv(self.path)
        
        except FileNotFoundError:
            self.log.error(f'File not found in {self.component} {self.path}' )
            raise
        
        except Exception as e:
            self.log.error(f'Unexpected error occurred while loading the {self.typ} in {self.component}, {e}')
            raise
