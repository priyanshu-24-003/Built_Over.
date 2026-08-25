import logging
import os



class Logy():

    """
    A class to Configure logging Entity for individual components to use accross the pipeline.
    """

    def __init__(self, logfilename, dirname):
        """
        logfilename : component.log for which the logs will be saved.
        dirname : directory where logs are saved .

        constructor create the direname/logs directory are logfilename.log is stored.
        constructor also creates self.This_logy an object of logging.getLogger
        """

        self.filename = logfilename
        self.This_logy = logging.getLogger(self.filename)
        self.This_logy.setLevel('DEBUG')

        self.log_dir = f'./{dirname}/logs'
        os.makedirs(self.log_dir, exist_ok=True)


    def get_this_logy(self):
        """
        Method configures fomatter, console handler and filehandler to (self.This_logy : logging.getLogger)
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel('DEBUG')

        log_file_path = os.path.join(self.log_dir, self.filename)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel('DEBUG')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.This_logy.addHandler(console_handler)
        self.This_logy.addHandler(file_handler)

        return self.This_logy


if __name__ == "__main__":
    
    L_A = Logy('A.log', 'data3').get_this_logy()
    
    L_A.debug('entered A compo')

    L_B = Logy('B.log', 'data3').get_this_logy()
    
    L_B.debug('entered B compo')

    L_C = Logy('C.log', 'data3').get_this_logy()
    
    L_C.debug('entered C compo')

    