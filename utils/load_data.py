import pandas as pd


def load_myrealfooding_data(all=False):
    try:
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")

        if all:
            return train_df, test_df
        else:
            return train_df

    except Exception as ex:
        print(ex)


