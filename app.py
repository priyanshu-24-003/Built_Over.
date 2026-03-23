from live_testing.fetch_data import *
import pickle
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    
    test = pd.read_csv('./live_testing/live_test.csv')

    return test


def load_model():

     with open('./data/models/model.pkl', 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    get_sample_data()
    test = load_data()
    features = test.iloc[:, :-1]
    label = test.iloc[:,-1]
    print("data_that_we_have \n")
    print(test, '\n\n')
    model = load_model()
    print("report that we get using our model for prediction \n")
    prediction = model.predict(features)
    print('accuracy ', accuracy_score(label, prediction), '\n\n')
    
