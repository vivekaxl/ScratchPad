# @uthor: vivek.nair@lexisnexis.com

from __future__ import division
import pandas as pd
def clean_up(filename):
    df = pd.read_csv(filename)
    df_no_missing = df.dropna()

    # http://stackoverflow.com/questions/32011359/convert-categorical-data-in-pandas-dataframe
    cat_columns = df_no_missing.select_dtypes(['object']).columns
    for cat_column in cat_columns:
        df_no_missing[cat_column] = df_no_missing[cat_column].astype('category')
    df_no_missing[cat_columns] = df_no_missing[cat_columns].apply(lambda x: x.cat.codes)
    df_no_missing.to_csv("cleaned_segmentation.csv", index=False)


if __name__ == "__main__":
    filename = "segmentation.csv"
    clean_up(filename)