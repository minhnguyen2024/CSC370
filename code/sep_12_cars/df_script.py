import pandas as pd

#index is row
def main():
    dictionary = {
        "x": pd.Series([1,2,3], index=["a", "b", "c"]),
        "y": pd.Series([4,5,6], index=["a", "b", "c"]),
    }
    df = pd.DataFrame(dictionary)
    print(df)
    # print(df.iloc[0,1])
    # print(df.loc["a", "y"])
    # print(df.loc[:"b", :"y"])
    print(df.loc[:'c', :"x"])

main()