"""
    CSC 370: Quarterback ratings, linear models, information from data
    Group members: Even Sajtar, Minh Nguyen
"""
import pandas as pd
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
    P_VALUE_SIGNIFICANT_LEVEL = 0.05
    features_df = model.summary2().tables[1]
    significant_features_df = features_df[features_df["P>|t|"] < P_VALUE_SIGNIFICANT_LEVEL]
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
    X = sm.add_constant(X)
    Y1 = df.loc[:, passer_rating_name]
    Y2 = df.loc[:, espn_rating_name]

    # 1a.
    print("------Task 1------")
    passer_rating_arr = []
    for i in range(len(df)):
        data_row = X.iloc[i, :]
        passer_rating_arr.append(passer_rating(data_row))

    df["Passer Rating"] = passer_rating_arr
    # 1b.
    print(df.loc[:, ["Player", "Rate", "Passer Rating"]])
    print("---------------------")

    # 2a.
    print("------Task 2------")
    y_passer_rating = df.loc[:, ["Passer Rating"]]
    rate_model = sm.OLS(y_passer_rating, X).fit()
    print(f"2b - significant features from Passer Rating model: {get_significant_features_from_model(rate_model)}")
    """
    2b. and 2c.
    At 0.05 significant level for p-value,
    the significant features are: ['Cmp%', 'TD%', 'Int%', 'AY/A', 'NY/A', 'ANY/A']
    
    2d. 
    The significant features are the percentage of Cmp, TD and Int, which is
    different from the passer_rating formula since it uses raw values (CMD, TD, INT, ATT)
    """
    # print(rate_model.summary2())
    print("---------------------")

    # 3a and 3b
    print("------Task 3------")
    print(f"3b: {get_significant_features_from_model(rate_model)}")
    print("---------------------")

    #4
    print("------Task 4------")
    qbr_model = sm.OLS(Y2, X).fit()
    qbr_model_significant_features = get_significant_features_from_model(qbr_model)
    print(f"4a: {qbr_model_significant_features}")
    print("---------------------")
    """
        The significant contributors to QBR are ['Att', 'Cmp%', 'Y/C'], while there are up
        to 6 significant contributors ['Cmp%', 'TD%', 'Int%', 'AY/A', 'NY/A', 'ANY/A'] to Rate from Task 2.
        This means there QBR and Rate are being produced using different formula and therefore depend on different
        features.
    """

    #5
    print("------Task 5------")
    print(f"Significant features being used for prediction model: {qbr_model_significant_features}")
    x_significant = df.loc[:, qbr_model_significant_features]
    significant_model = sm.OLS(Y2, x_significant).fit()
    y_pred_significant = significant_model.predict(x_significant)
    df["QBR predict from significant"] = y_pred_significant
    mean_squared_error = sum((Y2 - y_pred_significant)**2)/len(y_pred_significant)
    rmse_sig = np.sqrt(mean_squared_error)

    y_pred_qbr = qbr_model.predict(X)
    df["QBR predict"] = y_pred_qbr  
    mean_squared_error = sum((Y2 - y_pred_qbr)**2)/len(y_pred_significant)
    rmse = np.sqrt(mean_squared_error)

    print(f"rmse: {rmse} rmse_sig: {rmse_sig}")
    
    print(df.loc[:, ["Rate", "Passer Rating", "QBR", "QBR predict", "QBR predict from significant"]])
    print("---------------------")

    """
        I compared the predictions from only using the significant variables with using all variables,
        and the result is the full model prediction produces better results since the rmse is lower that the 
        rmse from the model only using significant features. The result makes sense since the full model uses
        more control variables but the significant model also produces close results (because it eliminates insignificant
        variables)
    """

    #6
    print("------Task 6------")
    y_diff = Y2 - Y1
    df["y_diff"] = y_diff
    y_diff_model = sm.OLS(y_diff, X).fit()
    y_diff_model_significant = get_significant_features_from_model(y_diff_model)
    print(f"6a - significant features: {y_diff_model_significant}")
    y_diff_model_significant.append("y_diff")
    y_diff_model_significant.append("Player")
    print(df.loc[:, y_diff_model_significant].sort_values(by="y_diff", ascending=True).head())

    qbr_model_att = qbr_model.summary2().tables[1].loc["Att", "Coef."]
    qbr_model_yc = qbr_model.summary2().tables[1].loc["Y/C", "Coef."]
    print(f"qbr_model_att: {qbr_model_att}, qbr_model_yc: {qbr_model_yc}")
    rate_model_att = rate_model.summary2().tables[1].loc["Att", "Coef."]
    rate_model_yc = rate_model.summary2().tables[1].loc["Y/C", "Coef."]
    print(f"rate_model_att: {rate_model_att}, rate_model_yc: {rate_model_yc}")

    # print(f"qbr_model significant features: {get_significant_features_from_model(qbr_model)}")
    # print(f"rate_model significant features: {get_significant_features_from_model(rate_model)}")
    print("---------------------")
    """
        The difference between Y2 and Y1 can be explained through the different significant 
        features collected from the model, Att and Y/C
        There is a big difference the between the coefficients for Att and Y/C for each model individually, 
        which explains the difference between the dependent variables 
    """

main()
