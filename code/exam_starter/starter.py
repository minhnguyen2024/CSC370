"""
    CSC 370 Midterm Exam
    Name:  Minh T. Nguyen
    email: minhnguyen_2024@depauw.edu
"""
# you should not need any other imports than these
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal     # normal and uniform are random sample generators
from numpy.random import uniform

''' ##############################################
    P1
    Put P1 helper functions below this line
    ---------------------------------------'''

def difference_from_mean(pd_series):
    return pd_series - pd_series.mean()
    


def p1():
    # your code here
    df = pd.read_csv('./exam_data/p1/mtcars.csv')
    wt_series = df["wt"]
    print()
    diff_wt = difference_from_mean(wt_series)
    # plt.hist()
    plt.hist(diff_wt)
    plt.show()
    pass
#------------------------------------------------#

''' ##############################################
    P2
    Put P2 helper functions below this line
    ---------------------------------------'''
def p2():
    # your code here
    uniform_dist = uniform(low=0, high=100, size=1000)
    normal_dist = normal(loc=50, scale=10, size=1000)
    uniform_dist_over_75 = filter(lambda x: x > 75, uniform_dist)
    normal_dist_over_75 = filter(lambda x: x > 75, normal_dist)

    uniform_dist_over_75_list = []
    normal_dist_over_75_list = []
    for item in uniform_dist_over_75:
        uniform_dist_over_75_list.append(item)
    for item in normal_dist_over_75:
        normal_dist_over_75_list.append(item)

    print(f"% of x values over 75 in a uniform dist. is {len(uniform_dist_over_75_list)/len(uniform_dist)}")
    print(f"% of x values over 75 in a normal dist. is {len(normal_dist_over_75_list)/len(normal_dist)}")
    # plt.hist(uniform_dist)
    plt.hist(normal_dist)
    plt.axvline(x=75, color="red")
    # plt.show()
    pass
#------------------------------------------------#

''' ##############################################
    P3
    Put P3 helper functions below this line
    ---------------------------------------'''
def calc_rmse(actual, predict):
    mean_squared = sum((actual - predict)**2)/len(predict)
    return np.sqrt(mean_squared)


def p3():
    # your code here
    train_df = pd.read_csv('./exam_data/p3/train_p3.csv')
    y_train = train_df["y"]
    x_train = train_df["x"]

    test_df = pd.read_csv('./exam_data/p3/test_p3.csv')
    y_test = test_df["y"]
    x_test = test_df["x"]
    # series = pd.Series({'A': y_train, 'B': x_train, 'C': y_test, "D": x_test})

    # 3d.
    for poly_deg in range(1, 6):
        coeffs = np.polyfit(x_train, y_train, poly_deg)
        y_predict = np.polyval(coeffs, x_train)
        rmse = calc_rmse(actual=y_train, predict=y_predict)
        print(f"3d. degree: {poly_deg} train rmse: {rmse}")
    print("----------------------")
    # 3e.
    for poly_deg in range(1, 6):
        coeffs = np.polyfit(x_train, y_train, poly_deg)
        y_predict = np.polyval(coeffs, x_test)
        rmse = calc_rmse(actual=y_test, predict=y_predict)
        print(f"3e. degree: {poly_deg} train rmse: {rmse}")
    """
    For 3d, the polynomial degree of 5 yields the lowest RMSE while predicting from the x test date in 3e. 
    yields the lowest value at polynomial degree of 2.
    """
    pass
#------------------------------------------------#

''' ##############################################
    P4
    Put P4 helper functions below this line
    ---------------------------------------'''
def p4():
    
    # your code here
    red_df = pd.read_csv('./exam_data/p4/winequality-red.csv', sep=';')
    x_red_df = red_df.drop(columns=["quality"])
    x_red_df = sm.add_constant(x_red_df)

    y_quality_red = red_df.loc[:, ["quality"]]

    y_red_model = sm.OLS(y_quality_red, x_red_df).fit()
    features_red_df = y_red_model.summary2().tables[1]
    significant_features_red_model = features_red_df[features_red_df["P>|t|"] < 0.05]
    significant_features_red_model_list = list(significant_features_red_model.index)

    white_df = pd.read_csv('./exam_data/p4/winequality-white.csv', sep=';')
    x_white_df = white_df.drop(columns=["quality"])
    x_white_df = sm.add_constant(x_white_df)
    y_quality_white = white_df.loc[:, ["quality"]]
    y_white_model = sm.OLS(y_quality_white, x_white_df).fit()
    features_white_df = y_white_model.summary2().tables[1]
    significant_features_white_model = features_white_df[features_white_df["P>|t|"] < 0.05]
    significant_features_white_model_list = list(significant_features_white_model.index)

    common_features = list(np.intersect1d(significant_features_white_model_list, significant_features_red_model_list))
    print(f"The common significant features are: {common_features}")

    pass
#------------------------------------------------#

''' ##############################################
    P5
    Put P5 helper functions below this line
    ---------------------------------------'''
def p5():
    # your code here
    auto_df = pd.read_csv('./exam_data/p5/auto_mpg_train.csv')
    test_df = pd.read_csv('./exam_data/p5/auto_mpg_test.csv')
    # print(auto_df)
    # print(test_df)

    y_mpg = auto_df.loc[:, ["mpg"]]
    x_auto_df = auto_df.drop(columns=["mpg","car_name"])
    x_auto_df = sm.add_constant(x_auto_df)
    # print(x_auto_df.head())
    auto_df_model = sm.OLS(y_mpg, x_auto_df).fit()

    y_mpg_test_df = test_df.loc[:, ["mpg"]]
    x_test_df = test_df.drop(columns=["mpg","car_name"])
    x_test_df = sm.add_constant(x_test_df)

    y_test_df_predict = auto_df_model.predict(x_test_df)
    rmse = calc_rmse(actual=y_mpg_test_df, predict=y_test_df_predict)
    plt.scatter(y_mpg_test_df, y_test_df_predict)
    print(rmse)
    pass
#------------------------------------------------#

''' ##############################################
    Main
    Do not not modify other than to comment out a question's code.
    Complete each problem in the provided function above.
'''
def main():
    # p1()
    # p2()
    # p3()
    # p4()
    p5()

''' ##############################################
    call main
'''
main()
