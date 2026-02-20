import pytest

#importing pipeline components

import src.A_data_injestion as A

with open('data/logs/data_ingestion.log') as file:
    
    r = file.readlines()[-6:]
    print(r)

