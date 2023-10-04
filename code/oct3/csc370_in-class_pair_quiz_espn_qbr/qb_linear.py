"""
    CSC 370: Quarterback ratings, linear models, information from data
    Group members:__________________
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def main():
    df = pd.read_csv('qb2018_simple.csv')
    print(df)
    print(df.columns)

    # use these as your starting features.
    # visit here to learn more about each stat: 
    # https://www.pro-football-reference.com/years/2018/passing.htm, click on the Glossary link.
    features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Yds',
                'TD','TD%', 'Int', 'Int%', 'Lng', 'Y/A', 'AY/A',
                'Y/C', 'Y/G', 'Sk', 'Yds.1', 'NY/A', 'ANY/A', 'Sk%']

    # these are the names of your target values.
    passer_rating_name = 'Rate'
    espn_rating_name = 'QBR'

    # some useful subsets
    X = df.loc[:,features]
    Y1 = df.loc[:, passer_rating_name]
    Y2 = df.loc[:, espn_rating_name]

    # some example code, please erase before submission
    row_index = 0
    full_row = df.iloc[row_index, :]
    data_row = X.iloc[row_index,:]
    print(type(data_row))
    print(full_row['Player'], full_row['Tm']) # full_row contains all data
    print(data_row) # data row is only independent variables / features.
    
    model1 = sm.OLS(Y1, X).fit()
    print(model1.summary2())
    y_pred = model1.predict(X)


main()
