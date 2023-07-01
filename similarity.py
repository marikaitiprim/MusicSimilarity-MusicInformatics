import os
import pandas as pd 
from IPython.display import display
import audio_metadata as am
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm


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

    #delete unimportant columns
    df = df.drop('Unnamed: 0',axis='columns')
    df = df.drop('Unnamed: 0.1',axis='columns')
    df = df.drop('cluster',axis='columns')

    mydict = {}
    i=0
    for fileflac in os.listdir(datapath):                       #for every file in dataset
        metadata = am.load(os.path.join(datapath, fileflac))    #read file
        if str(metadata['tags'].title) in df.values :           #if song title is in the cluster
            audiofile = librosa.load(os.path.join(datapath, fileflac))
            y, sr = audiofile
            mfccfeature = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13, n_fft=2048, hop_length=400)
            mydict[str(metadata['tags'].title)] = mfccfeature
        i+=1
        name = str(metadata['tags'].title)
        if(i==2):
            break

    #cosine_similarity(mydict["['Sensation']"], mydict["['Rosebud March']"])
    # import required libraries
 
    # compute cosine similarity
    cosine = cosine_similarity(mydict["['What Is There To Say']"],mydict[name])
    #cosine = np.dot(mydict["['Sensation']"],mydict["['Rosebud March']"])/(norm(mydict["['Sensation']"], axis=1)*norm(mydict["['Rosebud March']"]))
    print("Cosine Similarity:\n", cosine)


clusterSimilarity(0,datapath)