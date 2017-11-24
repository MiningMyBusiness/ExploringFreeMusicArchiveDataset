## Title: Genre Prediction with k-NN with feature and label dimensionality reduction
##  This code attempts to predict the genre of songs by using their musical features extracted from the audio file
##  This is a multi-class, multi-label problem. It is meant as a first pass exploration of the dataset. Specifically,
##  to see if the feature space is at all indicative of genre.
##
## Author: Kiran Bhattacharyya (bhattacharyyakiran12@gmail.com)
##
## License: MIT License

# import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim import corpora, models
import time
import tensorflow as tf
import re

# define function to normalize features
def normalize_features(featureMat):
    dimMeans = featureMat.mean(0)
    dimStds = featureMat.std(0)
    normMat = featureMat - np.tile(dimMeans, [featureMat.shape[0], 1]) # subtract mean from matrix
    normMat = np.divide(normMat, np.tile(dimStds, [featureMat.shape[0], 1])) # divide by standard deviation
    return normMat

# define function to extract genre names from a string of integers
#   songGenres is an array with genre ID/s for each song as a string (these must be broken at commas, turned into integers, and then matched to a name)
#   allGenreNames is an array containing the names of all genres
#   allGenreIds is an array containing the IDs of all genres
def extract_song_genre_names(songGenres, allGenreNames, allGenreIds):
    songGenreNames = list() # create list to populate with genre names
    for genres in songGenres: # for each string in the list of genre ID strings
        commaIndx = [m.start() for m in re.finditer(',', genres)] # find the position of commas
        if len(genres) == 2: # genre only has two brackets (no genre)
            songGenreNames.append(['N/A']) # add N/A for genre name
        if len(commaIndx) == 0 and len(genres) > 2: # no commas in the string and only one genre
            genreId = int(genres[1:-1]) # get genre ID as an integer
            genreIndx = np.where(allGenreIds == genreId) # find index of genre ID
            genreName = allGenreNames[genreIndx[0][0]] # pull genre name from array
            songGenreNames.append([genreName]) # append genre name to master list
        if len(commaIndx) > 0 and len(genres) > 2: # if a song has more than one genre (at least one comma in the string)
            tempList = list() # create temporary list
            for x in range(0,len(commaIndx) + 1): # go through each comma index and append genres ignoring starting and ending brackets
                if x == 0:
                    oneGenreId = int(genres[1:commaIndx[x]])
                    oneGenreIndx = np.where(allGenreIds == oneGenreId)
                    oneGenreName = allGenreNames[oneGenreIndx[0][0]]
                elif x > 0 and x < len(commaIndx):
                    oneGenreId = int(genres[commaIndx[x-1]+1:commaIndx[x]])
                    oneGenreIndx = np.where(allGenreIds == oneGenreId)
                    oneGenreName = allGenreNames[oneGenreIndx[0][0]]
                elif x == len(commaIndx):
                    oneGenreId = int(genres[commaIndx[x-1]+1:-1])
                    oneGenreIndx = np.where(allGenreIds == oneGenreId)
                    oneGenreName = allGenreNames[oneGenreIndx[0][0]]
                tempList.append(oneGenreName) # append each genre to the temporary list
            songGenreNames.append(tempList) # append temporary list to full list
    return songGenreNames

# Define location of relevant files
featureDir = 'fma_metadata/features.csv' # features for each track
tracksDir = 'fma_metadata/tracks.csv' # metadata for every track
genreDir = 'fma_metadata/genres.csv' # genre key from ID to string

####################################################
#### SONG FEATURE PROCESSING ####

# Get features of songs
myFeatures = pd.read_csv(featureDir, low_memory=False) # load data
featureMat = myFeatures.values # turn to np matrix
onlyFeats = featureMat[3:,1:].astype(float) # extract just features w/o feature labels
onlyFeats_norm = normalize_features(onlyFeats) # Normalize feature set data

# feature Parameters
num_of_songs = onlyFeats_norm.shape[0]
num_of_features = onlyFeats_norm.shape[1]

# Apply dimensionality reduction on features
mypca = PCA(n_components=155) # create a pca operation which keeps 155 components
mypca.fit(onlyFeats_norm) # fit pca to feature matrix
smallFeats = mypca.transform(onlyFeats_norm) # transform feature matrix into lower dimension space

# redifine reduced feature number
num_of_features = smallFeats.shape[1]

#######################################
#### SONG GENRE (LABEL) PROCESSING ####

# Get the top genre for all songs
myLabels = pd.read_csv(tracksDir, low_memory=False) # load data
labelMat = myLabels.values # turn into np matrix
topGenreIndx = np.where(labelMat[0,:] == 'genres_all') # find column with top genre info for songs
songGenres = labelMat[2:,topGenreIndx[0][0]] # get song genres
num_of_songs = songGenres.shape[0]

# Get number of genres and genre names
genreNames = pd.read_csv(genreDir) # load in file with genre info
genreMat = genreNames.values # extract values from the data frame
allGenreName = genreMat[:,3] # get the names of genres
allGenreIds = genreMat[:,0] # get all genre ids
num_of_genres = allGenreIds.shape[0] + 1 # the total number of genres and plus 1 if a song has no genre listed
allGenreNameWithNoGenre = np.array(['long string with a lot of stuff']*num_of_genres) # create a new array to populate with all genre names
allGenreNameWithNoGenre[:(num_of_genres - 1)] = allGenreName # populate the first part with names of genres
allGenreNameWithNoGenre[num_of_genres-1] = 'N/A' # the non-genre class (some songs have no genre)

# Extract genre names from genre IDs as numbers from string
songGenreNames = extract_song_genre_names(songGenres, allGenreName, allGenreIds)

########################################
#### TOPIC MODELLING OF SONG GENRES ####

# Create dictionary from list of lists of song genre names
dictionary = corpora.Dictionary(songGenreNames)

# From dictionary create a bag-of-words corpus
corpus = [dictionary.doc2bow(genreNames) for genreNames in songGenreNames]

# Create LDA model with corpus and with fewer topics
num_of_topics = 70 # number of topics to reduce label space
myLdaModel = models.ldamodel.LdaModel(corpus, num_topics=num_of_topics, id2word = dictionary, passes=10)

# Convert corpus to LDA topics
corpus_LDA = myLdaModel[corpus]

# Create a binary matrix of genre topics
topicBinFeat = np.zeros([len(corpus_LDA), num_of_topics])
for i in range(0, len(corpus_LDA)):
    topics = corpus_LDA[i]
    for j in range(0, len(topics)):
        topicIndex = topics[j][0]
        topicBinFeat[i, topicIndex] = 1

###########################################################
#### PREDICT GENRE WITH k-NEAREST NEIGHBOR ####

trainProp = 0.95 # proportion of data for training (a large proportion is used to ensure full feature space is represented)
num_of_training = round(num_of_songs*trainProp) # number of songs for training
num_of_ks = 10 # number of k's to try for k-nearest neighbor tests
num_of_tests = 5 # number of times each nearest neighbor test will be run with randomized training and testing groups
fullAccMetric = np.zeros([num_of_ks,smallFeats[(num_of_training+1):].shape[0]]) # create a variable to store one entire set of accuracy metrics for each k-NN
numOfTrueLabels = np.zeros([num_of_ks, smallFeats[(num_of_training+1):].shape[0]]) # variable that will store the number of true genre labels for a test song
numOfPredLabels = np.zeros([num_of_ks, smallFeats[(num_of_training+1):].shape[0]]) # variable that will store the number of predicted genre labels for a test song
accMetric = np.zeros([num_of_ks, num_of_tests]) # create variable to store average of accuracy metric for tests

# Break into training and testing sets
for j in range(0,num_of_ks):
    myK = j + 1 # my number of neighbors (plus 1 to account for indexing starting with 0)
    for testNum in range(0,num_of_tests):
        print('For k-value ', str(myK), ' running test number ', str(testNum + 1))
        randOrder = np.random.permutation(num_of_songs) # create a random permutation of indices
        smallFeats = smallFeats[randOrder] # order sets randomly
        topicBinFeat = topicBinFeat[randOrder]
        trX = smallFeats[0:num_of_training,:] # break data into training and testing sets
        trY = topicBinFeat[0:num_of_training,:]
        teX = smallFeats[(num_of_training+1):,:]
        teY = topicBinFeat[(num_of_training+1):,:]

        # tf Graph input
        xtr = tf.placeholder("float", [None, num_of_features])
        xte = tf.placeholder("float", [num_of_features])

        # Nearest Neighbor calculation using L1 Distance
        # Calculate L1 Distance and make negative so small distances are closer to zero and large distances are very negative
        distance = tf.negative(tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1))
        # Prediction: Get min distance index (Nearest neighbors)
        values, myK_indices = tf.nn.top_k(distance, k=myK, sorted=False) # get the k least negative values (lowest distances)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            sess.run(init)

            # loop over test datas
            for i in range(len(teX)):
                # Get nearest neighbor
                nn_indices = sess.run(myK_indices, feed_dict={xtr: trX, xte: teX[i, :]}) # run predictor
                # Get nearest neighbor class label and compare it to its true label
                nnLabel = np.round(np.mean(trY[nn_indices,:], axis=0) + 0.01) # nearest neighbor label estimate (ensure 0.5 rounds to 1)
                trueLabel = teY[i,:] # true label
                # Calculate Jaccard index
                labelInt = np.sum(nnLabel*trueLabel) # intersection of predicted and true labels
                labelUni = np.sum(np.ceil((nnLabel + trueLabel)/2)) # union of predicted and true labels
                jaccDist = labelInt/labelUni # compute Jaccard index
                fullAccMetric[j,i] = jaccDist # save jaccard index
                numOfTrueLabels[j,i] = np.sum(trueLabel) # save the number of true genre labels
                numOfPredLabels[j,i] = np.sum(nnLabel) # save the number of predicted labels
            accMetric[j,testNum] = np.mean(fullAccMetric[j,:]) # compute and save mean jaccard index

        sess.close()

#################################################
#### SAVE DATA ####

# Save average accuracy metric for k-NN results for each trial for each k-NN
np.savetxt('Results/JaccMetric_kNN_featAndLabelDimReduce.csv', accMetric, delimiter=",")

# Save accuracy metric for every test point for every k-NN for one trial
np.savetxt('Results/fullJaccMetric_kNN_featAndLabelDimReduce.csv', fullAccMetric, delimiter=",")

# Save number of true labels for every test point for every k-NN for one trial
np.savetxt('Results/numOfTrueLabels_kNN_featAndLabelDimReduce.csv', numOfTrueLabels, delimiter=",")

# Save number of predicted labels for every test point...
np.savetxt('Results/numOfPredLabels_kNN_featAndLabelDimReduce.csv', numOfTrueLabels, delimiter=",")

#################################################
#### PLOT RESULTS ####

# Visualize Jaccard index for k-NNs
avgAccPerK = np.mean(accMetric, axis = 1) # find mean accuracy for all trials for each k-NN
stdAccPerK = np.std(accMetric, axis = 1) # find the standard deviation of the error
myKs = np.arange(num_of_ks) + 1 # array of k values
plt.errorbar(myKs, avgAccPerK, yerr=stdAccPerK, fmt='--o') # plot with standard deviation as error bars
plt.xlabel('Number of nearest neighbors')
plt.ylabel('Average Jaccard index')
plt.title('Performance of k-NN for with feature and genre dimensional reduction')
plt.savefig('Figures/kNNGenrePredictor_featAndLabelDimReduce.png')
plt.clf()
plt.close()

# Visualize histograms of Jaccard indices for each k-NN trial
for myK in range(0,num_of_ks):
    plt.hist(fullAccMetric[myK,:])
    plt.xlabel('Jaccard index')
    plt.ylabel('Number of predictions')
    myTitle = 'Histogram of Jaccard indices for k = ' + str(myK+1)
    plt.title(myTitle)
    myFigName = 'Figures/JaccIndxHistforK_featAndLabelDimReduce_' + str(myK+1) + '.png'
    plt.savefig(myFigName)
    plt.clf()
    plt.close()

# Visualize if the number of genre labels influences Jaccard index
for myK in range(0,num_of_ks):
    plt.plot(fullAccMetric[myK,:], numOfTrueLabels[myK,:], "o")
    plt.xlabel('Jaccard index')
    plt.ylabel('True number of genre labels for songs')
    myTitle = 'Jaccard index Relation to Number of True Labels k = ' + str(myK+1)
    plt.title(myTitle)
    myFigName = 'Figures/JaccIndxVsTrueLabelsforK_featAndLabelDimReduce_' + str(myK+1) + '.png'
    plt.savefig(myFigName)
    plt.clf()
    plt.close()

# Visualize if the number of predicted labels vs true labels
for myK in range(0,num_of_ks):
    plt.plot(numOfTrueLabels[myK,:], numOfPredLabels[myK,:], "o")
    plt.ylabel('Predicted number of topics for songs')
    plt.xlabel('True number of topics for songs')
    myTitle = 'Number of True vs. Predicted Labels k = ' + str(myK+1)
    plt.title(myTitle)
    myFigName = 'Figures/TrueVsPredLabelsforK_featAndLabelDimReduce_' + str(myK+1) + '.png'
    plt.savefig(myFigName)
    plt.clf()
    plt.close()
