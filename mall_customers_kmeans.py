# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score

def replace_non_numeric(df):
	df["Genre"] = df["Genre"].apply(lambda sex: 0 if sex == "Male" else 1)
	return df

mall = pd.read_csv("Datasets/Mall_Customers.csv")
mall.drop(["CustomerID"], axis=1, inplace=True)
replace_non_numeric(mall)

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(mall)

print(mall)
print(k_means.labels_)

score = silhouette_score(mall, k_means.labels_, metric='euclidean')

print("")
print('Silhouette Score: %.3f' % score)
