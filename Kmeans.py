from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()
iris = pd.DataFrame(iris.data, columns=iris['feature_names'])

sse = {}
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i).fit(iris)
    iris["clusters"] = kmeans.labels_
    sse[i] = kmeans.inertia_
#Elbow method
figure= plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()