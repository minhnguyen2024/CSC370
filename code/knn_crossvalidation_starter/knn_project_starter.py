"""
    A program to predict wine quality.
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import random
# new imports (sklearn = scikit-learn)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def create_fold_indices(df, n_folds):
    indices_shuffled = df.index.tolist()
    random.shuffle(indices_shuffled)
    set_size = round(len(indices_shuffled) / n_folds)
    folds = []
    for i in range(0, len(indices_shuffled), set_size):
        folds.append(indices_shuffled[i: i + set_size])
    return folds

def train_test_index_split(fold_indices, current_test_index):
    shallow_fold_indices = fold_indices[:]
    test_indices = shallow_fold_indices.pop(current_test_index)
    train_indices = np.concatenate(shallow_fold_indices).tolist()
    return train_indices, test_indices


def main():
    red_df = pd.read_csv('./data/winequality-red.csv', sep=';')
    np.random.seed(42)
    fold_indices = create_fold_indices(red_df, 10)
    acc_scores_df = pd.DataFrame(index=range(10), columns=range(1, 50, 2))
    for k_value in range(1, 50, 2):
        acc_scores = []
        for i in range(10):
            train_indices, test_indices = train_test_index_split(fold_indices, i)
            train_df = red_df.iloc[train_indices, :]
            test_df = red_df.iloc[test_indices, :]

            y_train = train_df['quality']
            x_train = train_df.drop(columns=['quality'])

            # standardize train data
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train_standardized = scaler.transform(x_train)

            # train knn model
            knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
            # knn_classifier.fit(x_train_standardized, y_train)
            # fit w/o std
            knn_classifier.fit(x_train, y_train)

            # validation
            y_test = test_df['quality']
            x_test = test_df.drop(columns=['quality'])

            # standardize test data
            x_test_standardized = scaler.transform(x_test)
            # y_pred = knn_classifier.predict(x_test_standardized)
            # predict w/o std
            y_pred = knn_classifier.predict(x_test)

            acc_score = accuracy_score(y_pred, y_test)
            acc_scores.append(acc_score)
        acc_scores_df[k_value] = acc_scores
    acc_scores_df.plot.box()
    plt.show()

main()
