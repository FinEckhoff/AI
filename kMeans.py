import sklearn.cluster
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

model = KMeans()
scaler = StandardScaler()
def init(clusters):
    global model
    model = KMeans(init="random", n_clusters=clusters, random_state=0, n_init="auto")

def train(X_train) -> sklearn.cluster._kmeans:
    #scaled_features = scaler.fit_transform(X_train)
    scaled_features = X_train
    model.fit(scaled_features)
    return model

def eval(X_train):
    silhouette_coefficients = []
    scaled_features = X_train
    #scaled_features = scaler.fit_transform(X_train)
    for k in range(2, 11):
        model = KMeans(init="random", n_clusters=k, random_state=0, n_init="auto")
        model.fit(scaled_features)
        score = silhouette_score(scaled_features, model.labels_)
        silhouette_coefficients.append(score)
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()






