from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
import logging
import yaml
import numpy as np
import mlflow

#importing helper module
from helper.Loader import Load
from helper.Saver import Savy
from helper.Logster import Logy

LC= Logy('C_Feature_selection.log','data')
logger = LC.get_this_logy()

def remove_duplicate_unordered(lst):
    """
    removes duplicated combos of features.
    """
    lst = [[{a[0], a[1], a[2]}] for a in lst if a[1] != a[2]]
    sets = {frozenset(x[0]) for x in lst}
    final_features_combos = [list(y) for y in sets]
    return final_features_combos

def check_column_scores(cols, df):
    """
    
    K-Fold-Cross-Validation technique with Xtrain[cols]
    
    It is not validating the model but the features [cols] "combo" that are used to train the model.  
    """

    LR = LogisticRegression(max_iter = 5000)
    
    df2 = df.copy()
    y = df['Species']
    X = df2.drop(["Species"], axis = 1)

    return cross_val_score(LR, X[cols], y, cv = 5).mean()



def FeatureSelection(df):
    """
    Methods to select a combination 3 most important features 

    1. First it creats all combos
    2. Passes every combo to training set to check_column_scores to find mean cv score it has produced.
    3. Combo that produced best mean cv score gets selected at the end
    """

    try:
        logger.debug('feature selection started ')
        quals = ["Island", "Sex"]

        quants = ["Culmen Length (mm)", "Culmen Depth (mm)",
                "Flipper Length (mm)", "Body Mass (g)"]

        combos = [[qual]+[quant1]+[quant2] for qual in quals for quant1 in quants for quant2 in quants if quant1 != quant2]

        #I hadn't remove_duplicate_unordered , feature selection time would have doubled.
        combos = remove_duplicate_unordered(combos)

        cv_scores = []
        best_cv_score = -np.inf

        for combo in combos:

            #passing training set with current combo for k-fold-cross-validation
            score = check_column_scores(combo, df)
            cv_scores.append(score)
            
            if cv_scores[-1] > best_cv_score:
                best_cv_score = cv_scores[-1]
                best_combo = combo

        logger.debug(f"Best Feature selected {best_combo} which produces CV score: " + str(best_cv_score),)

        return best_combo
    except:
        logger.error('error occured while selecting best featueres')
        raise
    pass


def main():
    try:

        train_data = Load('./data/interim/train.csv', 'df', logger, __file__).load_it()
        test_data = Load('./data/interim/test.csv', 'df', logger, __file__).load_it()

        BestFeatures = FeatureSelection(train_data) + ['Species']

        train_df = train_data[BestFeatures]
        test_df = test_data[BestFeatures]

        #Saving data
        save_path = os.path.join("./data", "processed")
        Savy(save_path, 'df', logger, __file__, train_df, test_df).save_it()

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")
 


if __name__ == '__main__':
    main()