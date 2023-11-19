"""
    A program to predict wine quality.
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

def distance(a, b):
    return np.sqrt(np.sum((b-a)**2))

def main():
    red_df = pd.read_csv('./data/winequality-red.csv', sep=';')

    # create a train test split
    red_train = red_df.sample(frac=0.8, random_state=42)
    #rest of df net train set
    red_test = red_df.drop(red_train.index)

    # separate target labels from features
    # x and y for train set
    X_features = red_train.drop(columns=['quality'])
    Y_labels = red_train['quality']

    # use a scaler to encode the scaling parameters
    scaler = StandardScaler()
    scaler.fit(X_features)      # "fit" the scaler to the training data.

    X_features_standardized = scaler.transform(X_features)

    pd.DataFrame(X_features_standardized).plot.box()
    # pd.DataFrame(X_features).plot.box()
    # plt.show()

    # "train" the classifier
    for k in range(1, 21, 3):
        k_value = k
        knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
        knn_classifier.fit(X_features_standardized, Y_labels)

        X_test = red_test.drop(columns=['quality'])
        Y_test = red_test['quality']

        X_test_standardized = scaler.transform(X_test) # scale using training set parameters

        y_pred = knn_classifier.predict(X_test_standardized)
        # print(y_pred, Y_test)

        accuracy = accuracy_score(y_pred, Y_test)
        print(k, accuracy)

    # PCA
    # pca = PCA()
    # pca.fit(X_features_standardized)
    # transformed_data = pca.transform(X_features_standardized)
    #print(transformed_data)

    # print(transformed_data.shape)
    # row_index = random.sample(range(transformed_data.shape[0]), 100)
    # print(row_index)

    # labels = Y_labels.iloc[row_index]
    # print(labels)

    # plt.scatter(transformed_data[row_index, 0], transformed_data[row_index, 1])
    # plt.title('PCA Transformed Data')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()
    


    
main()