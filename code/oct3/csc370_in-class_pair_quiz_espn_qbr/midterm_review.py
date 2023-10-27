import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

#tuple
tuple = ("a", "b", "c", 0, 9, True)
# print(tuple[5])

val1, val2, val3, val4, val5, val6 = tuple
# print(val2)

#pandas series
series_from_dict = pd.Series({'A': 10, 'B': 20, 'C': 30})
# print(series_from_dict)
series_with_index = pd.Series([9, 8, 7], index=["x", "y", "z"])
# print(f"mean: {series_with_index.mean()}")
series_without_index = pd.Series([1, 2, 3]) #index will be automatically generated
# print(series_without_index)
series_tuple_with_index = pd.Series([("Minh", "21"), ("Wendy", "21")], index=["person1", "person2"])
series_tuple_with_index_1 = pd.Series({'person1': ("Minh", 43.99), 'person2': ("Keegan", 32.22), 'person3': ("Cole", 23.54)})

series = pd.Series(["s", "f", 5, 6, 4, True, 3, False, "hello"])
# print(f"from series[3] to series[5]: {series[2,6]}")
# print(f"from all but last 2 elements: {series[:-2]}")
# print(f"from series[3] but last 2 elements: {series[3:-2]}")

df = pd.read_csv('qb2018_simple.csv')
y_var = df.loc[:, ["Rate"]]
x_var = df.loc[:, ["TD"]]

# x_var_idx = df.iloc[0:11, 0:1]
# y_var_idx = df.iloc[0:11, 1:2]


# print(x_var, y_var)
# plt.scatter(x_var, y_var)
# plt.scatter(x_var_idx, y_var_idx)
plt.xlabel("TD")
plt.ylabel("Rate")
# plt.show()
