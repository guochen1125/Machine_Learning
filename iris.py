import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_data["data"], iris_data["target"], random_state=0
)

iris_pd = pd.DataFrame(X_train, columns=iris_data.feature_names)
pd.plotting.scatter_matrix(
    iris_pd,
    c=y_train,
    figsize=(12, 12),
    marker="o",
    hist_kwds={"bins": 20},
    s=60,
    alpha=0.8,
)
plt.savefig("./imgs/scatter_matrix.png")


knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(knn.score(X_test, y_test))
