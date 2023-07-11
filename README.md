# Music Similarity in jazz dataset
Music similarity is a well-known MIR task in the machine learning field. In this project, we try to examine the similarities between different songs of the same genre. 
We choose a large dataset of audio files (.flac) with different songs of the same genre - here we chose jazz together with its subgenres - and we extract certain features. 
Then, we perform clustering methods, such as K-means and Agglomerative to split our dataset in clusters. From these clusters, we can already see similarities between the songs, such as tone key (minor/major),
loudness, tempo and duration. The next step is to examine each cluster seperately. By computing the mfccs and chromagrams of each song in a cluster, we can calculate the distances between each song and display a 
similarity matrix, where according to the color of each cell, we understand how much similar 2 songs are. For more results and observations, check the report. 
