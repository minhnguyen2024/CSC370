"""
    A program to explore and visualize some car data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from scipy import stats
# import statsmodels.api as sm

import random

def main():
    df = pd.read_csv('./data/mtcars.csv')
    x = df['mpg']
    y = df['hp']

    # print(np.corrcoef(x, y))
    # correlation = stats.pearsonr(x,y)[0]
    # print(correlation)
    # correlation = stats.pearsonr(df['mpg'],y)[0]
    # print(correlation)

    df = pd.read_csv('./data/mtcars.csv')
    x = df['mpg']
    y = df['hp']

    # perform a linear fit for the data, first degree polynomial
    coeffs = np.polyfit(x, y, 1)            # fit polynomial
    newX = np.linspace(min(x),max(x),100)    # x values for plotting (create a regular intervals)
    predicted = np.polyval(coeffs, newX)    # get polynomial value at each x

    plt.scatter(df['mpg'], df['hp'])        # plot scatter plot
    plt.plot(newX, predicted, color="red")  # plot the fit line
    plt.show()                              # show the plot

    # # fit a linear model in 2D, plane of best bit, uses scikit-learn
    # x = df[['mpg','wt']]                # X, attributes / features for prediction
    # y = df['hp']                        # y, target values of interest
    # regressor = LinearRegression()      
    # # use scikit-learn LinearRegression for multiple dependent variables
    # regressor.fit(x, y)                 # peform the fit

    # fit a linear model in 2D, plane of best bit, uses statsmodels
    # x = df[['mpg','wt']]                # X, attributes / features for prediction
    # y = df['hp']                        # y, target values of interest
    # x = sm.add_constant(x)              # tell the model to have an intercept
    # model = sm.OLS(y, x).fit()          # perforn the fitting / training
    # print(model.summary2())             # print summary information
    # p_values = model.summary2().tables[1]['P>|t|']
    # print(p_values)
    

    #print(p_value)

    #plt.hist(df['mpg'])
    # plt.scatter(df['mpg'], df['hp'])
    # plt.plot(regressor.predict(x))
    # plt.show()

    # values = [(random.random() - 0.5)*3 for i in range(100)]
    # values = np.cumsum(values)
    # plt.plot(values)
    # plt.show()

    
    # plt.boxplot([df['mpg'], df['hp']], labels=['mpg', 'hp'])
    # plt.show()

main()
