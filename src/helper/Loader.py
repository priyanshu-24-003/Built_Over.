import pandas as pd
import pickle 
import logging
from pathlib import Path



def get_filename(path_str: str) -> str:
    return Path(path_str).name


class Load():
    """
    A class to Load Different types of data in the Pipeline.
    """

    def __init__(self,file_path:str, file_type:str, logger:logging.getLogger, component):
        """
        file_type:['df', 'model']
        logger: Configured logging entity that saves logs for the component.
        component: Component of the Pipeline (A, B, C, D, E. (__File__) at the time of execution)
        """
        self.path = file_path
        self.typ = file_type
        self.log = logger
        self.component = get_filename(component)


    def load_it(self,):
        """
        This method finally loads self.path file of self.typ in the self.component and also logs it.
        : for self.typ = 'modle' : open(self.path, 'rb')
        : for self.typ = 'df' : pandas.read_csv(self.path)
        """

        self.log.debug(f'started Data_loading inside component {self.component}')
        try:
            if self.typ == 'model':

                with open(self.path, 'rb') as file:
                    file = pickle.load(file)
                    self.log.debug(f'model Loaded successfully inside component {self.component}')
                    return file
                
            elif self.typ == 'df':
                file = pd.read_csv(self.path)
                self.log.debug(f'Data Loaded successfully inside component {self.component}')
                return file
        
        except FileNotFoundError:
            msg = f'File {self.path} not found in {self.component} '
            self.log.error(msg)
            raise msg
        
        except Exception as e:
            msg = f'Unexpected error occurred while loading the {self.typ} in {self.component}, {e}'
            self.log.error(msg)
            raise msg
