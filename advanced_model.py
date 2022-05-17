from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from load_data import load_myrealfooding_data
from data_ops import clean_data, feature_engineering
import pickle
import xgboost as xgb


def main():
    try:
        # Load the data
        train = load_myrealfooding_data()
        print(train.shape)

        # Drop useless columns and Nan values
        train = clean_data(train, columns=["id"], drop_na=True)
        print(train.shape)

        # Create new variables and modify others
        X, y = feature_engineering(train)

        # Let's split the data into train, validation and test sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        # Model creation with hyperparameter optimization and cross validation
        param_dist = {
            'n_estimators': [100, 250, 500],
            'max_depth': [6, 9, 12],
            'subsample': [0.9, 1.0],
            'colsample_bytree': [0.9, 1.0],
        }

        xgb_clf = xgb.XGBClassifier()
        clf = GridSearchCV(xgb_clf, param_dist, cv=3, scoring='neg_log_loss', n_jobs=-1)
        clf.fit(X_train, y_train)

        # Predict values from validation set and compute accuracy
        preds_xgb = clf.predict(X_val)

        accuracy_xgb = float(np.sum(preds_xgb == y_val))/y_val.shape[0]
        print('Accuracy de XGBoost en validation: ', accuracy_xgb)

        # Store the model to make predictions with the test dataset
        model_path = "models/advanced_model_xgb.pkl"
        print(f"Saving model into {model_path}")
        pickle.dump(clf, open(model_path, "wb"))

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
