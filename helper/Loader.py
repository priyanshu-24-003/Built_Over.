import pandas as pd
import pickle 
import logging

class Load():

    def __init__(self,file_path:str, file_type:str, logger:logging.getLogger):
        self.path = file_path
        self.typ = file_type
        self.log = logger

    def load_it(self,):
        self.log.debug(f'started Data_loading inside component {__file__}')
        try:
            if self.typ == 'model':

                with open(self.path, 'rb') as file:
                    self.log.debug(f'Data Loaded successfully inside component {__file__}')
                    return pickle.load(file)
                
            elif self.typ == 'df':
                self.log.debug(f'Data Loaded successfully inside component {__file__}')
                return pd.read_csv(self.path)
        
        except FileNotFoundError:
            self.log.error('File not found: %s', self.path)
            raise
        
        except Exception as e:
            self.log.error('Unexpected error occurred while loading the model: %s', e)
            raise
