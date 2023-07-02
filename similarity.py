import os
import pandas as pd 
from IPython.display import display
import audio_metadata as am
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


datapath = '/media/marikaiti/ASUS_PROG-FILES/music_similarity_jazz_dataset'

#select the number of the cluster to be examined
def clusterSimilarity(clusternum, datapath):
    if clusternum == 0:
        #read file
        df = pd.read_csv('cluster0.csv')
    if clusternum == 1:
        #read file
        df = pd.read_csv('cluster1.csv')
    if clusternum == 2:
        #read file
        df = pd.read_csv('cluster2.csv')
    if clusternum == 3:
        #read file
        df = pd.read_csv('cluster3.csv')

    mydict = {}
    mylist = []
    songlist = []
    i=0
    for fileflac in os.listdir(datapath):                       #for every file in dataset
        metadata = am.load(os.path.join(datapath, fileflac))    #read file
        if str(metadata['tags'].title) in df.values :           #if song title is in the cluster
            audiofile = librosa.load(os.path.join(datapath, fileflac))
            y, sr = audiofile
            mfccfeature = librosa.feature.mfcc(y=y,sr=sr)
            songlist.append(str(metadata['tags'].title))
            #mydict[str(metadata['tags'].title)] = mfccfeature.T
            mylist.append(mfccfeature.T)
            i+=1
        if(i==11):
            break

    # Calculate the similarity matrix in batches
    num_mfccs = len(mylist)
    similarity_matrix = np.zeros((num_mfccs, num_mfccs))

    for i in range(num_mfccs):
        for j in range(num_mfccs):
            if i == j:
                similarity_matrix[i, j] = 1.0       #same audio song
            else:
                distances = cdist(mylist[i], mylist[j], metric='euclidean')
                similarity_matrix[i, j] = np.mean(distances)

    print(similarity_matrix)
    
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

    # Adding a colorbar
    plt.colorbar()

    # Adding titles for each block
    titles = [songlist[0], songlist[1], songlist[2], songlist[3], songlist[4], songlist[5], songlist[6], songlist[7], songlist[8], songlist[9], songlist[10]]  # Add your own titles here
    num_titles = len(titles)
    #plt.xticks(np.arange(num_titles), titles)
    plt.yticks(np.arange(num_titles), titles)

    # Displaying the plot
    plt.show()

clusterSimilarity(0,datapath)