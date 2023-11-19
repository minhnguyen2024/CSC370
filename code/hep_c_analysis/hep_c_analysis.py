"""
    A program to explore hepatitis C data.
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import random
# new imports (sklearn = scikit-learn)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# 'Category', 'Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'
def main():
    hepc_df = pd.read_csv('hcvdat0.csv', index_col=0)
    # print(hepc_df.columns)

    train_df = hepc_df.sample(frac=.8, random_state=0)
    test_df = hepc_df.drop(index=train_df.index)

    # Train data on train_df
    y_labels = train_df['Category']
    x_data = train_df.drop(columns=['Category', 'Sex'])
    # print(y_labels.unique())

    # clean data, replace NaN with median
    print(x_data.iloc[x_data.index == 122])
    medians = x_data.median()
    x_data = x_data.fillna(medians)
    print(medians["CHOL"])
    print(x_data.iloc[x_data.index == 122])

    # standardize data with scaler
    # x_data.plot.box()
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # knn model
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_data, y_labels)
    # pd.DataFrame(x_data).plot.box()
    # plt.show()

    # Applying to test set
    y_test = test_df["Category"]
    x_test = test_df.drop(columns=['Category', 'Sex'])

    # clean
    x_test = x_test.fillna(medians)

    # standardize:
    # cannot fit on test set, only transform
    x_test = scaler.transform(x_test)

    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cmat = confusion_matrix(y_test, y_pred)
    # print(f"accuracy score: {acc}")
    # print(cmat)

    pca = PCA()
    pca.fit(x_data)
    transformed_data = pca.transform(x_data)
    print(y_labels.unique())
    c0 = y_labels[y_labels == '0=Blood Donor']
    c1 = y_labels[y_labels == '1=Hepatitis']
    c2 = y_labels[y_labels == '3=Cirrhosis']
    c3 = y_labels[y_labels == '0s=suspect Blood Donor']
    c4 = y_labels[y_labels == '2=Fibrosis']

    targets = ['0=Blood Donor', '1=Hepatitis', '3=Cirrhosis', '0s=suspect Blood Donor', '2=Fibrosis']
    colors = ['r', 'g', 'b', 'y', 'p']
    
    for target, color in zip(targets, colors):
        indices_to_keep = y_labels == target
        plt.scatter(pd.DataFrame(transformed_data).loc[indices_to_keep, 'PC1'], 
                    pd.DataFrame(transformed_data).loc[indices_to_keep, 'PC2'], c=color)
    plt.show()


main()