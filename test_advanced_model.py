import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from load_data import load_myrealfooding_data
from data_ops import clean_data, feature_engineering
import pickle
import xgboost as xgb

# open a file, where you stored the pickled data
file = open('models/advanced_model_xgb.pkl', 'rb')

# dump information to that file
clf = pickle.load(file)
clf = clf.best_estimator_

# close the file
file.close()

# Load the data
train, test = load_myrealfooding_data(all=True)

# Drop useless columns and Nan values
test = clean_data(test, columns=["id"], drop_na=True)

# Create new variables and modify others
X, y = feature_engineering(test)


preds_xgb = clf.predict(X)

accuracy_xgb = float(np.sum(preds_xgb == y)) / y.shape[0]
print('Accuracy de XGBoost en test: ', accuracy_xgb)
