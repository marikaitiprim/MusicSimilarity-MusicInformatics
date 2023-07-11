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


    mfcclist = []
    songlist = []
    chromalist = []
    i=0

    for fileflac in os.listdir(datapath):                       #for every file in dataset

        metadata = am.load(os.path.join(datapath, fileflac))    #read file

        if str(metadata['tags'].title) in df.values :           #if song title is in the cluster

            audiofile = librosa.load(os.path.join(datapath, fileflac))
            y, sr = audiofile
            mfccfeature = librosa.feature.mfcc(y=y,sr=sr)       #find mfcc
            chromafeature = librosa.feature.chroma_cqt(y=y,sr=sr)   #find chroma

            songlist.append(str(metadata['tags'].title))
            mfcclist.append(mfccfeature.T)
            chromalist.append(chromafeature.T)
            i+=1

        if(i==11):                                              #use the first 10 songs to be found
            break

    #initialize similarity matrices
    num_mfcc = len(mfcclist)
    num_chroma = len(chromalist)
    similarity_matrix_mfcc = np.zeros((num_mfcc, num_mfcc))
    similarity_matrix_chroma = np.zeros((num_chroma, num_chroma))

    for i in range(num_mfcc):                   #num_mfcc == num_chroma
        for j in range(num_mfcc):
            if i == j:
                similarity_matrix_mfcc[i, j] = 1.0       #same audio song
                similarity_matrix_chroma[i, j] = 1.0
            else:
                distance_mfcc = cdist(mfcclist[i], mfcclist[j], metric='euclidean') #distance for mfccs
                distance_chroma = cdist(chromalist[i], chromalist[j], metric='euclidean')   #distance for chroma
                similarity_matrix_mfcc[i, j] = np.mean(distance_mfcc)               #construct similarity matrix for mfcc
                similarity_matrix_chroma[i, j] = np.mean(distance_chroma)           #construct similarity matrix for chroma

    print(similarity_matrix_mfcc)
    print(similarity_matrix_chroma)

    fig = plt.figure(1)
    fig.suptitle('MFCC similarity matrix')
    plt.imshow(similarity_matrix_mfcc, cmap='hot', interpolation='nearest')

    # Adding a colorbar
    plt.colorbar()

    # Adding titles for each block
    titles = [songlist[0], songlist[1], songlist[2], songlist[3], songlist[4], songlist[5], songlist[6], songlist[7], songlist[8], songlist[9], songlist[10]]  # Add your own titles here
    num_titles = len(titles)
    plt.yticks(np.arange(num_titles), titles)

    # Displaying the plot
    plt.show()


    fig = plt.figure(2)
    fig.suptitle('Chroma similarity matrix')
    plt.imshow(similarity_matrix_chroma, cmap='hot', interpolation='nearest')

    # Adding a colorbar
    plt.colorbar()

    # Adding titles for each block
    titles = [songlist[0], songlist[1], songlist[2], songlist[3], songlist[4], songlist[5], songlist[6], songlist[7], songlist[8], songlist[9], songlist[10]]  # Add your own titles here
    num_titles = len(titles)
    plt.yticks(np.arange(num_titles), titles)

    # Displaying the plot
    plt.show()

clusterSimilarity(0,datapath)
