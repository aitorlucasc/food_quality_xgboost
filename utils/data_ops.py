import pandas as pd


def clean_data(df, columns, drop_na=True):
    try:
        print(f"Droping columns: {columns}")
        df.drop(columns=columns, inplace=True)
        if drop_na:
            print(f"Deleting null values from data")
            df.dropna(axis=0, inplace=True)

        return df

    except Exception as ex:
        print(ex)


def add_not_liquid(df):
    try:
        df.loc[df["is_liquid"] == 0, "not_liquid"] = 1
        df.loc[df["is_liquid"] != 0, "not_liquid"] = 0
        df["not_liquid"] = df["not_liquid"].astype(int)

        return df

    except Exception as ex:
        print(ex)


def to_categorical(df, col):
    try:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
        return df
    except Exception as ex:
        print(ex)


def clean_ingredients_list(x):
    try:
        chars_to_remove = "[]'"
        for value in chars_to_remove:
            x = x.replace(value, "")
        x = x.split(",")
        return x
    except Exception as ex:
        print(ex)


def feature_engineering(df):
    try:
        df = add_not_liquid(df)
        df = to_categorical(df, "category")
        df["ingredients_ordered"] = df["ingredients_ordered"].apply(clean_ingredients_list)
        df = df.explode('ingredients_ordered').reset_index(drop=True)
        df = to_categorical(df, "ingredients_ordered")
        y = df["target"]
        df.drop(columns=["target"], inplace=True)
        return df, y
    except Exception as ex:
        print(ex)
