

#this file fetches random_sample of the test_data any new data can be introduced in future
import pandas as pd



def get_sample_data():
    df = pd.read_csv('data/processed/test.csv')

    sample_test = df.sample(30)

    sample_test.to_csv('./live_testing/live_test.csv', index=False)



if __name__ == "__main__":
    get_sample_data()