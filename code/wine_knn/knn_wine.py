"""
    A program to predict wine quality.
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def distance(a, b):
    return np.sqrt(np.sum((b-a)**2))

def main():
    red_df = pd.read_csv('./data/winequality-red.csv', sep=';')

    # print(red_df.head())
    # print(red_df.shape)

    # create a train test split
    red_train = red_df.sample(frac=0.8, random_state=42)
    red_test = red_df.drop(red_train.index)

    # separate target labels from features
    X_features = red_train.drop(columns=['quality'])
    Y_labels = red_train['quality']
    
    # show box plot
    X_features.plot.box()
    # plt.show()

    # calculate the mean and standard deviation
    X_mean = X_features.mean()
    X_sd = X_features.std()

    X_features_standardized = (X_features - X_mean)/X_sd
    X_features_standardized.plot.box()
    plt.show()

    X = X_features_standardized
    # X = X_features
    Y = Y_labels

    test_index = 0

    X_test = red_test.drop(columns=['quality'])
    Y_test = red_test['quality']

    X_test = (X_test - X_mean)/X_sd

    # get number of examples (rows, wines to predict for)
    N = X_test.shape[0]
    correct = 0

    for test_index in range(N):
        #test_index = 2
        # get vector from test set
        test_X_vector = X_test.iloc[test_index,:]
        test_Y_label = Y_test.iloc[test_index]

        # print(test_X_vector)
        # print(test_X_vector.index)

        # calcualte distances to trianing examples
        distances = X.apply(lambda x: distance(x, test_X_vector),axis=1)
        
        # choose training set label of nearest neighbor
        # 1-nn
        # predicted_label = Y.iloc[np.argmin(distances)]
        
        # 3-nn
        predicted_labels = Y.iloc[np.argsort(distances)[:3]]
        # print('mode:', predicted_labels.mode()[0])
        predicted_label = predicted_labels.mode()[0]

        # compare results
        #print(predicted_label)
        # print(predicted_label, test_Y_label, "hit" if predicted_label == test_Y_label else "miss")
        if predicted_label == test_Y_label:
            correct += 1

        if test_index % 100 == 99:
            print("current accuracy:", correct/(test_index+1))
    
    accuracy = correct/N
    print(accuracy)







main()