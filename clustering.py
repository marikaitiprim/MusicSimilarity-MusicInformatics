import pandas as pd 
from IPython.display import display
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from kneed import KneeLocator
import numpy as np

from sklearn.cluster import AgglomerativeClustering

le = preprocessing.LabelEncoder()

#read file
df = pd.read_csv('music_dataset.csv')

#endcoding non numerical data
df['scale'] = le.fit_transform(df['scale'])     #0-->major, 1-->minor

#remove columns that are not numeral values
dfcopy = df.copy()

dfcopy.drop('song title', inplace=True, axis=1)
dfcopy.drop('artist', inplace=True, axis=1)
dfcopy.drop('album title', inplace=True, axis=1)
dfcopy.drop('year', inplace=True, axis=1)

#standardize values to [0,1]
scaler = StandardScaler()
dfstd = scaler.fit_transform(dfcopy)

#PCA
pca = PCA(n_components=3)
pca_features = pca.fit_transform(dfstd)
evr = pca.explained_variance_ratio_

pca_df = pd.DataFrame(data=pca_features, columns=['Col1', 'Col2', 'Col3'])

plt.figure(1)     #80% of 4 to find the appropriate n_components
plt.plot(evr.cumsum(), marker='o')
plt.title('Number of features to use in kmeans')
plt.show()

x = pca_features[:,0]
y = pca_features[:,1]
z = pca_features[:,2]

fig = plt.figure(2)            #pca 3d projection
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.title('PCA')
plt.show()

#elbow method - find optimum k for kmeans
max_clusters = 10
distortions = []

for i in range(1,max_clusters):
    kmeans_pca = KMeans(n_clusters=i)
    kmeans_pca.fit(pca_df)
    distortions.append(kmeans_pca.inertia_)

nclusters = KneeLocator([i for i in range(1,max_clusters)],distortions,curve='convex', direction='decreasing').knee
print('Optimal number of clusters', nclusters)

fig = plt.figure(3)            #elbow plot
plt.plot(range(1,max_clusters), distortions, marker='o', linestyle='--')
plt.title('Elbow method')
plt.show()

#kmeans
# kmeans_pca = KMeans(n_clusters=nclusters)
# kmeans_pca.fit(pca_df)
# df['cluster'] = kmeans_pca.labels_

# fig = plt.figure(4)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x,y,z, c=kmeans_pca.labels_)
# plt.title('Kmeans')
# plt.show()

#agglomerative
agglo_pca = AgglomerativeClustering(n_clusters=nclusters)
agglo_pca.fit(pca_df)
df['cluster'] = agglo_pca.labels_

#display(df)

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, c=agglo_pca.labels_)
plt.title('agglo')
plt.show()

#testing each cluster
def featuresByCluster(df, feature, cluster):
    curcluster = df.loc[df['cluster'] == cluster]
    y = curcluster[feature]
    x = [i for i in range(len(y))]
    plt.bar(x,y)
    plt.ylabel(feature.title())
    if (feature == 'scale'):
        return
    plt.hlines(np.mean(df[feature]), 0, len(y))

#cluster 0
fig = plt.figure(5)
fig.suptitle('Cluster 0')
plt.subplot(2,2,1)
featuresByCluster(df,'duration',0)

plt.subplot(2,2,2)
featuresByCluster(df,'bpm',0)

plt.subplot(2,2,3)
featuresByCluster(df,'scale',0)

plt.subplot(2,2,4)
featuresByCluster(df,'loudness',0)

plt.show()

df0 = df.loc[df['cluster'] == 0].copy()
df0.to_csv('cluster0.csv')

#cluster 1
fig = plt.figure(6)
fig.suptitle('Cluster 1')
plt.subplot(2,2,1)
featuresByCluster(df,'duration',1)

plt.subplot(2,2,2)
featuresByCluster(df,'bpm',1)

plt.subplot(2,2,3)
featuresByCluster(df,'scale',1)

plt.subplot(2,2,4)
featuresByCluster(df,'loudness',1)

plt.show()

df1 = df.loc[df['cluster'] == 1].copy()
df1.to_csv('cluster1.csv')

#cluster 2
fig = plt.figure(7)
fig.suptitle('Cluster 2')
plt.subplot(2,2,1)
featuresByCluster(df,'duration',2)

plt.subplot(2,2,2)
featuresByCluster(df,'bpm',2)

plt.subplot(2,2,3)
featuresByCluster(df,'scale',2)

plt.subplot(2,2,4)
featuresByCluster(df,'loudness',2)

plt.show()

df2 = df.loc[df['cluster'] == 2].copy()
df2.to_csv('cluster2.csv')

#cluster 3
fig = plt.figure(8)
fig.suptitle('Cluster 3')
plt.subplot(2,2,1)
featuresByCluster(df,'duration',3)

plt.subplot(2,2,2)
featuresByCluster(df,'bpm',3)

plt.subplot(2,2,3)
featuresByCluster(df,'scale',3)

plt.subplot(2,2,4)
featuresByCluster(df,'loudness',3)

plt.show()

df3 = df.loc[df['cluster'] == 3].copy()
df3.to_csv('cluster3.csv')


print('Most popular artists in each cluster')
print('Cluster 0:')
display(df0['artist'].value_counts().to_frame())
print('Cluster 1:')
display(df1['artist'].value_counts().to_frame())
print('Cluster 2:')
display(df2['artist'].value_counts().to_frame())
print('Cluster 3:')
display(df3['artist'].value_counts().to_frame())