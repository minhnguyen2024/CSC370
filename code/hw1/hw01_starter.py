"""
    A program to import data form file and calculate the mean using pandas
    and visualize the distribution using a histogram.
    adapted from an assignment by Steve Bogaerts.
"""
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

def main():
    ages = getAgeList()
    print(ages)
    print(calc_mean(ages))
    plt.hist(ages)
    plt.show()


def getAgeList(numRows = None):
    """Your description here ...
    b.
        The read_csv() function will parse data from a comma-separated values (.csv) file to a DataFrame, which is a data structure
        in Python similar to a table/spreadsheet.
        1st parameter is the path to csv file, telling Python where the data file is in the project folder
        2nd param (delimiter) indicates the character which separates each column in a row of data
        3rd param (header): whether the csv file contains headers or not (usually the first line of the csv file)
        4th param(names) are column names/labels
        5th param(engine): which programing language will be used to execute.
        6th param(nrows) is the number of rows will be read from the csv file, nrows=None means
        reading the whole file unless indicated otherwise
    c.
        origData is a DataFrame object, and it has a method loc(), which extracts columns of data given rows and columns.
        origData.loc[:, 'age'].tolist() will select all rows in the DataFrame (the stand alone : operator means selecting all), but only
        from the "age" column. The getAgeList() function will return a list of ages from every data entry.
    """
    colNames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    origData = pd.read_csv("data/adult.data", delimiter=", ", header=None, names=colNames, engine='python', nrows=numRows)
    return origData.loc[:, 'age'].tolist()


def calc_mean(arr: List[int]) -> float:
    return sum(arr)/len(arr)


main()

