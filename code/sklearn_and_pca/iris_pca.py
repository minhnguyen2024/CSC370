import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    iris_train = pd.read_csv("./data/train_iris.csv")
    iris_test = pd.read_csv("./data/test_iris.csv")

    features = ["sepal_length","sepal_width","petal_length","petal_width"]
    target = ["species"]

    x_train = iris_train.loc[:, features]
    y_train = iris_train.loc[:, target]
    print(y_train.head())

    scaler = StandardScaler()
    x_train_stardardized = scaler.fit_transform(x_train)
    
    pca = PCA(.95)


    principal_components = pca.fit_transform(x_train)

    principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    final_df = pd.concat([principal_df, y_train], axis=1)
    print(pca.n_components_)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.set_title("PCA", fontsize=20)

    targets = ['setosa', 'versicolor', 'virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indices_to_keep = final_df['species'] == target
        ax.scatter(final_df.loc[indices_to_keep, 'PC1'], final_df.loc[indices_to_keep, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    # plt.show()


main()