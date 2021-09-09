from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import numpy as np
#读取数据
X = np.loadtxt("2.txt",delimiter='\t')
#层次分类
Z = linkage(X, method='ward', metric='euclidean')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()