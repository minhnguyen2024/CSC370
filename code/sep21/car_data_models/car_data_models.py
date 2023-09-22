"""
    A program to explore and visualize some car data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

#call plt.show() after function call
def plot_curve(x, coeffs, x_label="", y_label=""):
    curve_resolution = 100
    plot_x = np.linspace(min(x), max(x), curve_resolution)
    plot_y = np.polyval(coeffs, plot_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(plot_x,plot_y,color="green")

def create_model(x, y, polynomial_degree):
    coeffs = np.polyfit(x, y, polynomial_degree)
    values = np.polyval(coeffs, x)
    mean_squared_error = sum((values - y)**2)/len(y)
    rmse = np.sqrt(mean_squared_error)
    plot_curve(x, coeffs, "wt", "mpg")
    return rmse

def main():
    """
        TASKS:
        1) Explore different values of polynomial_degree with your partner.
            How does the error change are you use larger and larger degree
            polynomials?
            What do you see when you use a degree 12 polynomial?
        2) Write a function that will plot your fitted model. This function should take
            min_x, max_x, and the coefficients list.
        3) Once your plot function is done, write a loop to explore the larger and larger models.
            Start at 1 (linear) and go up to 15. Does the error always decrease?
            Do these larger models seem like good fits? Not neccessary
        4) Comment out the above solutions and work on a new task.
            Compare a degree 1 and degree 2 model fit to mpg as a function of wt.
            This models miles per gallon (mpg) as a function of weight. mpg = f(wt)
            Which model gives the lowest error?
            Plot the scatter plot of wt vs mpg. Also plot both models in the same window.
    """
    df = pd.read_csv('./data/mtcars.csv')
    # df.plot.box()
    # plt.show()

    # x = df['hp']
    # y = df['qsec']
    # plt.scatter(x, y)
    # plt.xlabel("Horse Power")
    # plt.ylabel("quarter mile speed (seconds)")
    # plt.show()

    # the fit
    # polynomial_degree = 1
    # coeffs = np.polyfit(x, y, polynomial_degree)
    # the predictions
    # values = np.polyval(coeffs, x)
    # the estimated error / noise, rmse = root mean squared error
    # mean_squared_error = sum((values - y)**2)/len(y)
    # rmse = np.sqrt(mean_squared_error)
    # print(rmse)

    # plot the curve of the fitted model
    # turn this into a function

    # curve_resolution = 100
    # plot_x = np.linspace(min(x),max(x), curve_resolution)
    # plot_y = np.polyval(coeffs, plot_x)
    # plt.plot(plot_x,plot_y,color="green")
    # plt.show()


    rmse_arr = []
    # for polynomial_degree in range(1, 25):
    #     coeffs = np.polyfit(x, y, polynomial_degree)
    #     values = np.polyval(coeffs, x)
    #     mean_squared_error = sum((values - y)**2)/len(y)
    #     rmse = np.sqrt(mean_squared_error)
    #     rmse_arr.append(rmse)
    # print(f"rmse_arr {rmse_arr}")
    
    # plot_curve(x, coeffs)
    # plt.show()



    # uncomment after doing tasks 1-3.
    x = df['wt']
    y = df['mpg']
    # plt.xlabel("Weight")
    # plt.ylabel("Miles per Hour")
    plt.scatter(x, y)
    rmse_arr.append(create_model(x, y, 1))
    rmse_arr.append(create_model(x, y, 2))
    print(rmse_arr)
    plt.show()
    coeffs = np.polyfit(x, y, 2)
    # plot_curve(x, coeffs, "Weight", "Miles per hour")
    

    


main()
