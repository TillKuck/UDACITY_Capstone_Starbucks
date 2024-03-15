import pandas as pd
import numpy as np

def extract_value(df, dict_column, dict_key):
    """
    extracts values from dictionary column
    ----------
    Parameters:
    - df: dataframe that holds dictionary_column
    - dict_column: column of dataframe that is in dictionary format
    ----------
    Return: extracted value
    """
    if len(df[dict_column]) > 0:
        return df[dict_column][0].get(dict_key)
    else:
        return np.nan()


def transform_dict_col_to_col(df, dict_column, dict_key):
    """
    Transforms dictionary column to own column based on dictionary key
    ----------
    Parameters:
    - df: dataframe that holds dictionary_column
    - dictionary_column: column of dataframe that is in dictionary format
    - dict_key: dictionary key whose values shall be extracted to new column
    ----------
    Return: dataframe with new column
    """
    df[dict_key] = df[dict_column].apply(extract_value(df, dict_column, dict_key))

    return df

