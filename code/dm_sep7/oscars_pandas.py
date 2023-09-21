import pandas as pd
import matplotlib.pyplot as plt
def main():
    filename = "data/oscar_data.csv"
    df = pd.read_csv(filename)

    print(df.columns)

    categories = df["category"].unique()
    categories.sort()
    print(categories[:5])

main()