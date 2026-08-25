
### git clone : https://github.com/priyanshu-24-003/Built_Over..git


### env create :

        for ubuntu systems:
             python3 -m venv overenv
             source overenv/bin/activate
        
        for windows : DIY :)

### install deps : pip install -r requirements.txt

### terminal 1:
    #activate env
    
    mlflow ui

### terminal 2:
    activate env

    dvc init

    dvc repro

### Check following to see if it succeeded:

    1. 'raw', 'interim', 'processed', 'logs', 'model', reports folder with thier contents exit inside data/ dir

    2. mlartifacts/ dir exists with latest experiments

    3. data/logs : check individual logs : and see if "ERROR" key word is their; if not then congratulations.
