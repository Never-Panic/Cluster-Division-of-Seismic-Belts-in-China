import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import cluster
import collections

X = np.loadtxt("2.txt",delimiter='\t')
#聚类
num=8
clst=cluster.AgglomerativeClustering(n_clusters=num,linkage='ward')
predicted_lables=clst.fit_predict(X)

c = collections.Counter(predicted_lables)

for i in range(num):
    print("\n第",i+1,"簇")
    print("数量:",c[i])
    print('样本:')
    one_cluster = X[predicted_lables== i]
    print(one_cluster[0])

plt.figure()
#原始数据散点图，按照分类查看
plt.scatter(X[:, 0], X[:, 1], c=predicted_lables)
plt.show()