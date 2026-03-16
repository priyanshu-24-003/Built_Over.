import pytest

#importing pipeline components

import A_data_injestion as A
import helper as H

# def test_check():
#     assert sum([1,2, 3]) == 6, 'just a silly test'
#     pass

def Log_check(filename, recentlogN):
    with open(filename) as file:
        
        r = file.readlines()[-recentlogN:]
        allone = ''.join(r)
        if 'ERROR' in allone:
            raise "ERROR IN DATA_INJESTION"

def test_Data_ingestion():
    try:        
        Log_check('data/logs/data_ingestion.log', -6)
    except Exception as e:
        raise e
    
def test_data_preparation():
    
    try:        
        Log_check('data/logs/B_preparation.log', -6)
    except Exception as e:
        raise e
    

def test_feature_selection():
    
    try:        
        Log_check('data/logs/C_Feature_selection.log', -6)
    except Exception as e:
        raise e
    


def test_Evaluation():
    
    try:        
        Log_check('data/logs/D_Model_Training.log', -7)
    except Exception as e:
        raise e
    


def test_Model_training():
    
    try:        
        Log_check('data/logs/E_model_evaluation.log', -7)
    except Exception as e:

        raise e
    
