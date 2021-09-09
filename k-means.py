import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import collections

X = np.loadtxt("10y5-10.txt",delimiter='\t')

#设置k
k = 10
km = KMeans(n_clusters=k).fit(X)

#计算轮廓系数
score = metrics.silhouette_score(X, km.labels_)
print("轮廓系数:",score)
print("CH index:",metrics.calinski_harabasz_score(X, km.labels_))

#输出每簇的中心点、个数、所有样本点
c = collections.Counter(km.labels_)

for i in range(k):
    print("\n第",i+1,"簇")
    print("中心点:",km.cluster_centers_[i])
    print("数量:",c[i])
    print('所有样本:')
    one_cluster = X[km.labels_ == i]
    print(one_cluster)

plt.figure()
#原始数据散点图，按照分类查看
plt.scatter(X[:, 0], X[:, 1], c=km.labels_)
centroids = km.cluster_centers_
#重心红色X进行突出
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
plt.show()


