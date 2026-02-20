import pytest

#importing pipeline components

import A_data_injestion as A
import helper as H

# def test_check():
#     assert sum([1,2, 3]) == 6, 'just a silly test'
#     pass

def test_A():
    try:        
        with open('data/logs/data_ingestion.log') as file:
            
            r = file.readlines()[-6:]
            allone = ''.join(r)
            if 'ERROR' in allone:
                raise "ERROR IN DATA_INJESTION"
    except Exception as e:
        raise e