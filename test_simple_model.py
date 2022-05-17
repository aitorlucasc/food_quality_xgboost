import numpy as np
from load_data import load_myrealfooding_data
from data_ops import clean_data
import pickle

# open a file, where you stored the pickled data
file = open('models/simple_model_xgb.pkl', 'rb')

# dump information to that file
clf = pickle.load(file)

# close the file
file.close()

# Load the data
train, test = load_myrealfooding_data(all=True)

# Drop useless columns and Nan values
test = clean_data(test, columns=["id", "category", "ingredients_ordered"], drop_na=True)

# Create new variables and modify others
y_test = test["target"]
test.drop(columns=["target"], inplace=True)
preds_xgb = clf.predict(test)

accuracy_xgb = float(np.sum(preds_xgb == y_test)) / y_test.shape[0]
print('Accuracy de XGBoost en test: ', accuracy_xgb)
