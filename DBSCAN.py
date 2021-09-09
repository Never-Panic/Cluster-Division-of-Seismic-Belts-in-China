import numpy as np  
import sklearn.cluster as skc  
from sklearn import metrics   
import matplotlib.pyplot as plt  
import collections

X = np.loadtxt("3.txt",delimiter='\t')

db = skc.DBSCAN(eps=1.0, min_samples=3).fit(X) #DBSCAN聚类方法 
labels = db.labels_  #和X同一个维度，labels对应索引序号的值为其所在簇的序号。若簇编号为-1，表示为噪声

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
print('分簇的数目:',n_clusters_)
print("轮廓系数:",metrics.silhouette_score(X, labels))
print("CH index:",metrics.calinski_harabasz_score(X, labels))
raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
print('噪声比:',raito)

#输出每簇的个数、所有样本点
c = collections.Counter(labels)
for i in range(n_clusters_):
    one_cluster = X[labels == i]
    if(c[i]>50):
        print("\n第",i+1,"簇")
        print("数量:",c[i])
        print('样本:')
        print(one_cluster[0])
    plt.plot(one_cluster[:,0],one_cluster[:,1],'x')



#输出离散点
# print("\n离散点")
# discrete = X[labels == -1]
# print("数量:",len(discrete))
# print(discrete)
# plt.plot(discrete[:,0],discrete[:,1],'.')

plt.show()