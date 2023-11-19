"""
    A program to explore features of quarterbacks.
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import random
# new imports
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def main():
    qb_df = pd.read_csv("./data/qb2018_simple.csv")
    print(qb_df.shape)

    features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Yds', 'TD',
                'TD%', 'Int', 'Int%', 'Lng', 'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Sk', 'Yds.1',
                'NY/A', 'ANY/A', 'Sk%']
    
    X_data = qb_df[features]
    Y_data = qb_df['QBR']

    scaler = StandardScaler()
    scaler.fit(X_data)

    X_data_standardized = scaler.transform(X_data)

    pca = PCA()
    pca.fit(X_data_standardized)
    transformed_data = pca.transform(X_data_standardized)

    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    for i in range(transformed_data.shape[0]):
        name = qb_df['Player'].iloc[i]
        plt.text(transformed_data[i,0],transformed_data[i,1],name)

    
    data = pd.DataFrame(X_data_standardized,columns=features)
    model = sm.OLS(Y_data, sm.add_constant(data)).fit()
    print(model.summary2())


    # print(qb_df)

    # print(pca.components_)
    # print(pca.components_.shape)
    loadings = pca.components_

    # Create a DataFrame for better visualization (optional)
    loadings_df = pd.DataFrame(loadings, columns=features)

    # Display the factor loadings
    print("Factor Loadings:")
    for i in range(5):
        print(loadings_df.iloc[i,:])
        print(features[np.abs(loadings_df.iloc[i,:]).argmax()])
        print("explained", pca.explained_variance_ratio_[i])

    plt.show()

main()
