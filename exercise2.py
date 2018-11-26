import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Problem:

    @staticmethod
    def add_hr(df):
        df['mean_hr'] = df['hit'].mean()
        return df

    def split_train_test(df, p):
        df_train, df_test = train_test_split(df, test_size=p)
        return df_train, df_test


json_string = '{"datetime":{"0":1528955662000,"1":1528959255000,"2":1528965487000,"3":1528966204000,"4":1528966289000,"5":1528971637000,"6":1528974438000,"7":1528975251000,"8":1528982200000,"9":1528992569000,"10":1528994282000},"hit":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":1,"6":1,"7":0,"8":1,"9":0,"10":1}}'
df_rfqs = pd.read_json(json_string)
df_feature = Problem.add_hr(df_rfqs)
print("DataFrame with feature added:")
print(df_feature)
print("-" * 40)
print("Train and test split, showing df_train:")
df_train, df_test = Problem.split_train_test(df_feature, 0.8)
print(df_train)