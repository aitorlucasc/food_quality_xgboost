from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from load_data import load_myrealfooding_data
from data_ops import clean_data
import pickle


def main():
    try:
        # Load the data
        train = load_myrealfooding_data()
        print(train.shape)

        # Drop useless columns and Nan values
        train = clean_data(train, columns=["id", "category", "ingredients_ordered"], drop_na=True)
        print(train.shape)

        # Let's split the data into train, validation and test sets
        X_train, X_val, y_train, y_val = train_test_split(train.iloc[:, :-1], train["target"], test_size=0.2)

        # Fit the model
        clf_xgb = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, seed=123)
        clf_xgb.fit(X_train, y_train)

        # Predict values from validation set and compute accuracy
        preds_xgb = clf_xgb.predict(X_val)

        accuracy_xgb = float(np.sum(preds_xgb == y_val))/y_val.shape[0]
        print('Accuracy de XGBoost en validation: ', accuracy_xgb)

        # Store the model to make predictions with the test dataset
        model_path = "models/simple_model_xgb.pkl"
        print(f"Saving model into {model_path}")
        pickle.dump(clf_xgb, open(model_path, "wb"))

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
