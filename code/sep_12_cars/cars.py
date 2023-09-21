"""
    A program to explore and visualize some car data
"""
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # df = pd.read_csv('./data/mtcars.csv')
    s = pd.Series([2,4,3,4], index=["w", "x", "y", "z"])
    # print(s)
    # print(s.iloc[2])
    # print(s.loc["w"])
    # print(s.index.get_loc("x"))
    h = pd.Series([4,5,6,7])
    print(h.loc[1:])
    print(h.iloc[1:])
main()
