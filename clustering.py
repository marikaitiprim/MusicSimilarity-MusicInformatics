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

le = preprocessing.LabelEncoder()

#read file
df = pd.read_csv('music_dataset.csv')

#endcoding non numerical data
df['scale'] = le.fit_transform(df['scale'])

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
plt.plot(evr.cumsum())
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
kmeans_pca = KMeans(n_clusters=nclusters)
kmeans_pca.fit(pca_df)
df['cluster'] = kmeans_pca.labels_

#display(df)

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, c=kmeans_pca.labels_)
plt.title('Kmeans')
plt.show()


#testing each cluster
def featuresByCluster(df, feature, cluster):
    plt.figure(figsize=(10,8))
    curcluster = df.loc[df['cluster'] == cluster]
    y = curcluster[feature]
    x = [i for i in range(len(y))]
    plt.bar(x,y)
    plt.ylabel(feature.title())
    plt.hlines(np.mean(df[feature]), 0, len(y))
    plt.show()

# featuresByCluster(df,'duration',0)
# featuresByCluster(df,'bpm',0)
# featuresByCluster(df,'scale',0)
# featuresByCluster(df,'loudness',0)

# featuresByCluster(df,'duration',1)
# featuresByCluster(df,'bpm',1)
# featuresByCluster(df,'scale',1)
# featuresByCluster(df,'loudness',1)

# featuresByCluster(df,'duration',2)
# featuresByCluster(df,'bpm',2)
# featuresByCluster(df,'scale',2)
# featuresByCluster(df,'loudness',2)

# featuresByCluster(df,'duration',3)
# featuresByCluster(df,'bpm',3)
# featuresByCluster(df,'scale',3)
# featuresByCluster(df,'loudness',3)

# display(df.loc[df['cluster'] == 2])