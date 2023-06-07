import os
import pandas as pd 
from IPython.display import display
import audio_metadata as am
import librosa
import essentia
import essentia.standard as es
import soundfile as sf
import pyloudnorm as pyln


#create dataframe
df = pd.DataFrame(columns=['song title', 'artist', 'album title', 'year', 'duration'])             #create an empty dataframe

datapath = '/media/marikaiti/ASUS_PROG-FILES/music_similarity_jazz_dataset'

i = 0
bpm = []
scale = []
loudness = []
for fileflac in os.listdir(datapath):                       #for every file in dataset
    metadata = am.load(os.path.join(datapath, fileflac))                                         #read file
    df.loc[i] = [metadata['tags'].title,metadata['tags'].artist,metadata['tags'].album,metadata['tags'].date,metadata['streaminfo'].duration]       #insert row with metadata of .flac file
    i+=1

    #extract features
    audiofile = librosa.load(os.path.join(datapath, fileflac))
    y, sr = audiofile

    #extract bpm
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)        
    bpm.append(format(tempo))
    # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    #extract scale (major/minor)
    features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],rhythmStats=['mean', 'stdev'],tonalStats=['mean', 'stdev'])(os.path.join(datapath, fileflac))
    scale.append(features['tonal.key_edma.scale'])
    #print("Key/scale estimation:", features['tonal.key_edma.key'], features['tonal.key_edma.scale'])

    data, rate = sf.read(os.path.join(datapath, fileflac))
    meter = pyln.Meter(rate) 
    loud = meter.integrated_loudness(data)
    loudness.append(loud)
    # print(fileflac, loud)

df['bpm'] = bpm
df['scale'] = scale
df['loudness'] = loudness

display(df)
df.to_csv('music_dataset.csv')


