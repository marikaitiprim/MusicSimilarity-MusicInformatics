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
    i=0
    for fileflac in os.listdir(datapath):                       #for every file in dataset
        metadata = am.load(os.path.join(datapath, fileflac))    #read file
        if str(metadata['tags'].title) in df.values :           #if song title is in the cluster
            audiofile = librosa.load(os.path.join(datapath, fileflac))
            y, sr = audiofile
            mfccfeature = librosa.feature.mfcc(y=y,sr=sr)
            mydict[str(metadata['tags'].title)] = mfccfeature.T
            mylist.append(mfccfeature.T)
            i+=1
        if(i==7):
            break

    # Calculate the similarity matrix
    num_mfccs = len(mylist)
    similarity_matrix = np.zeros((num_mfccs, num_mfccs))

    # downsample_factor = 2

    # # Determine the new shape of the downsampled matrix
    # new_shape = (similarity_matrix.shape[0] // downsample_factor, similarity_matrix.shape[1] // downsample_factor)

    # # Reshape the matrix by grouping values into larger blocks
    # downsampled_matrix = similarity_matrix[:new_shape[0]*downsample_factor, :new_shape[1]*downsample_factor]
    # downsampled_matrix = downsampled_matrix.reshape(new_shape[0], downsample_factor, new_shape[1], downsample_factor)

    # # Average values within each block
    # downsampled_matrix = np.mean(downsampled_matrix, axis=(1, 3))

    # print(downsampled_matrix)

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

    # Adding labels
    plt.xlabel('MFCC2')
    plt.ylabel('MFCC1')

    # Displaying the plot
    plt.show()

clusterSimilarity(0,datapath)