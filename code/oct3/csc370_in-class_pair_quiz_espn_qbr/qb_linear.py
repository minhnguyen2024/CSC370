"""
    CSC 370: Quarterback ratings, linear models, information from data
    Group members: Even Sajtar, Minh Nguyen
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def passer_rating(data_row):
    ATT = data_row["Att"]
    CMP = data_row["Cmp"]
    YDS = data_row["Yds"]
    TD = data_row["TD"]
    INT = data_row["Int"]
    a = (CMP / ATT - 0.3) * 5
    b = (YDS / ATT - 3) * 0.25
    c = (TD / ATT) * 20
    d = 2.375 - (INT / ATT * 25)
    return ((a + b + c + d) / 6) * 100


def get_significant_features_from_model(model):
    features_df = model.summary2().tables[1]
    significant_features_df = features_df[features_df["P>|t|"] < 0.05]
    return list(significant_features_df.index)


def main():
    df = pd.read_csv('qb2018_simple.csv')
    # use these as your starting features.
    # visit here to learn more about each stat: 
    # https://www.pro-football-reference.com/years/2018/passing.htm, click on the Glossary link.
    features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Yds',
                'TD', 'TD%', 'Int', 'Int%', 'Lng', 'Y/A', 'AY/A',
                'Y/C', 'Y/G', 'Sk', 'Yds.1', 'NY/A', 'ANY/A', 'Sk%']

    # these are the names of your target values.
    passer_rating_name = 'Rate'
    espn_rating_name = 'QBR'
    # some useful subsets
    X = df.loc[:, features]
    Y1 = df.loc[:, passer_rating_name]
    Y2 = df.loc[:, espn_rating_name]

    # 1a.
    passer_rating_arr = []
    for i in range(len(df)):
        data_row = X.iloc[i, :]
        passer_rating_arr.append(passer_rating(data_row))

    df["Passer Rating"] = passer_rating_arr
    # 1b.
    print(df.loc[:, ["Player", "Rate", "Passer Rating"]])
    print("---------------------")

    # 2a.
    y_passer_rating = df.loc[:, ["Passer Rating"]]
    rate_model = sm.OLS(y_passer_rating, X).fit()
    """
    2b. and 2c.
    At 0.05 significant level for p-value,
    the significant features are: Cmp%, TD%, Int%, Y/A, AY/A, Y/C, NY/A, ANY/A
    
    2d. 
    The significant features are the percentage of Cmp, TD and Int, which is
    different from the passer_rating formula since it uses raw values (CMD, TD, INT, ATT)
    """
    print(rate_model.summary2())
    print("---------------------")

    # 3a and 3b
    print(f"3b: {get_significant_features_from_model(rate_model)}")
    print("---------------------")

    #4
    qbr_model = sm.OLS(Y2, X).fit()
    # print(qbr_model.summary2())
    print(f"4a: {get_significant_features_from_model(qbr_model)}")
    print("---------------------")
    """
        The only significant contributor to QBR is Y/C, while there are up
        to 7 significant contributors to Rate from Task 2.
        There is a correlation between QBR and Y/C, meaning that Y/C is one of the parameters
        involved in calculating QBR.
    """

    #5
    x_significant = df.loc[:, ["Y/C"]]
    significant_model = sm.OLS(Y2, x_significant).fit()
    y_pred_significant = significant_model.predict(x_significant)
    df["QBR predict from significant"] = y_pred_significant


    y_pred_qbr = qbr_model.predict(X)
    df["QBR predict"] = y_pred_qbr
    print(df.loc[:, ["QBR", "QBR predict", "QBR predict from significant"]])
    print("---------------------")

    """
        I compared the predictions from only using the significant variable (Y/C) with using all variables,
        and the result is the full model prediction produces better results since its prediction values
        are closer to QBR than only using Y/C. 
        My argument is that the variables other than Y/C are individually insignificant but jointly significant
        therefore they have more predictive power.
    """

    #6
    y_diff = Y2 - Y1
    df["y_diff"] = y_diff
    y_diff_model = sm.OLS(y_diff, X).fit()
    print(df.loc[:, ["Player", "QBR predict", "y_diff", "Y/C", "Cmp%", "TD%", "Int%", "Y/A", "AY/A", "NY/A", "ANY/A"]].sort_values(by="y_diff", ascending=True).head())
    print(f"6a: {get_significant_features_from_model(y_diff_model)}")
    print("---------------------")
    """
        While Y/C is the only statistically significant varible, 
        it does not have much predictive power as explained in Task 5.
    """

main()
