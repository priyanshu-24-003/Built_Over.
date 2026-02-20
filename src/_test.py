import pytest

#importing pipeline components

import A_data_injestion as A
import helper as H

# with open('data/logs/data_ingestion.log') as file:
    
#     r = file.readlines()[-6:]
#     print(r)

def test_check():
    assert sum([1,2, 3]) == 6, 'just a silly test'
    pass
