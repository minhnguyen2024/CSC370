'''
    This script explores distance measures and classification

    Task:
    1) Expland block 1 to classify all samples in the test set.
    2) Calculate the prediction accuracy.
        prediction accuracy = # of accuate predictions / # predictions.
    3) Calculate a confusion matrix for your predictions.
        https://en.wikipedia.org/wiki/Confusion_matrix
        A confusion matrix gives more context to the performance of a classifie.
        The confusion matrix is a table where the columns count predicted labels
        The rows count the actual labels. Correct prediction will line the diagonals.
    4) Try to expland the nearest neighbor classifier to a 3-nearest neighbor classifier.
        Compare your 3-nn to your 1-nn results in terms of both accuracy and confusion matrix.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def distance(a, b):
    return np.sqrt(np.sum((b-a)**2))

def separate_data_labels(dataframe, label_name):
    """
        Separate the data vectors from their label (class/category)
    """
    X = dataframe.drop(columns=label_name)
    Y = dataframe[label_name]
    return X, Y


def main():
    # load training and test data.
    train_df = pd.read_csv('./train_iris.csv')
    test_df = pd.read_csv('./test_iris.csv')

    # separate data matrix from classification lables. See function above.
    train_X, train_Y = separate_data_labels(train_df, 'species')
    test_X, test_Y = separate_data_labels(test_df, 'species')
    test_vector = test_X.iloc[0,:]

    # ----------- Block 1 ----------- #
    # begin classification on test set.
    correct_ct = 0
    predicted = []
    actual = []
    arr_cor_3nn = []
    for test_index in range(0, test_df.index.size):
        test_vector = test_X.iloc[test_index,:]
        # test_vector [4.9, 3.0, 1.4, 0.2]
        true_label = test_Y.iloc[test_index]
        # true_label [setosa]
        distances = train_X.apply(lambda x:distance(test_vector,x),1)
        # train_X is df of x variables from train set
        # lambda: test_vector = [4.9, 3.0, 1.4, 0.2], x = [6.8, 3.0, 5.5, 2.1]
        # distances is a list of the distance from each data point in the train set to x (test_vector)
        indices_3nn = np.argsort(distances)[:3]
        correct_ct_3nn = 0
        for idx in indices_3nn:
            if train_Y.iloc[idx] == true_label: correct_ct_3nn+=1
        arr_cor_3nn.append(correct_ct_3nn)
        index = np.argmin(distances)
        # basically find the index of the data point in the train set that is closest to x (test_vector)
        # aka find the closest point to x
        pred_label = train_Y.iloc[index]    
        # print('training set index:', index, ', predicted:', pred_label, ', actual:', true_label)
        predicted.append(pred_label)
        actual.append(true_label)
        if pred_label == true_label:
            correct_ct += 1
    
    print(f"1-nn accuracy: {correct_ct/test_df.index.size}")
    
    # for each 3nn how many of them gives the correct prediction from test_vector
    print(f"3nn accuracy: {sum(arr_cor_3nn) / (len(arr_cor_3nn)*3)}")
    labels = ["setosa", "versicolor", "virginica"]
    conf_matrix = [[0,0,0], 
                   [0,0,0], 
                   [0,0,0]]
    
    for i in range(len(actual)):
        conf_matrix[labels.index(predicted[i])][labels.index(actual[i])] += 1

    # print_matrix(3,3, conf_matrix)

    # print(confusion_matrix(actual, predicted))


def print_matrix(row_len, col_len, matrix):
    for row in range(row_len):
        print("--------")
        for col in range(col_len):
            print(matrix[row][col])

main()
