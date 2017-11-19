## Title: Genre Prediction with Multilayer Perceptron
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

# define function to extract genre IDs as numbers from string
#   songGenres is an array with genre ID/s for each song as a string (these must be broken at commas and turned into integers)
def extract_song_genre_ids(songGenres):
    songGenreIds = list() # create list to populate with genre ID/s integers
    for genres in songGenres: # for each string in the list of genre ID strings
        commaIndx = [m.start() for m in re.finditer(',', genres)] # find the position of commas
        if len(genres) == 2: # genre only has two brackets (no genre)
            songGenreIds.append([0]) # add 0 for genre Id
        if len(commaIndx) == 0 and len(genres) > 2: # no commas in the string and only one genre
            songGenreIds.append([int(genres[1:-1])]) # append the genre to the list
        if len(commaIndx) > 0 and len(genres) > 2: # if a song has more than one genre (at least one comma in the string)
            tempList = list() # create temporary list
            for x in range(0,len(commaIndx) + 1): # go through each comma index and append genres ignoring starting and ending brackets
                if x == 0:
                    oneGenre = int(genres[1:commaIndx[x]])
                elif x > 0 and x < len(commaIndx):
                    oneGenre = int(genres[commaIndx[x-1]+1:commaIndx[x]])
                elif x == len(commaIndx):
                    oneGenre = int(genres[commaIndx[x-1]+1:-1])
                tempList.append(oneGenre) # append each genre to the temporary list
            songGenreIds.append(tempList) # append temporary list to full list
    return songGenreIds

# Define function to organize genres of songs into a binary feature vector
#   songGenreIds is a list with genre ID/s for each song as an integers
#   allGenreIds is an array with genre ID/s, this is used to map to an index as genre IDs are not true indices
#   num_of_songs is the number of songs in the dataset
#   num_of_genres is the number of genres in the dataset (the highest genre ID is higher than this number since genre IDs are not true indices)
def genre_ids_to_binFeats(songGenreIds, allGenreIds, num_of_songs, num_of_genres):
    songGenreCode = np.zeros([num_of_songs, num_of_genres]) # for each song there will be a binary vector indicating which genres it belongs to
    for i in range(0, len(songGenreIds)): # for every song
        genreIds = songGenreIds[i] # get the genres
        temp_list = ([0]*num_of_genres) # create a list of zeros to update with song genres
        for j in range(0, len(genreIds)): # for each genre
            genreid = genreIds[j] # get one song genre
            if genreid > 0: # if genreid is not 0 (song has a genre)
                idToIndx = np.where(allGenreIds == genreid) # find genre ID
                idToIndx = idToIndx[0][0]
                temp_list[idToIndx] = 1
            else: # if genreid is zero (song has no genre)
                temp_list[-1] = 1
        temp_list = np.array(temp_list)
        songGenreCode[i,:] = temp_list # store binary genre label vector in main variable
    return songGenreCode

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

#######################################
#### SONG GENRE (LABEL) PROCESSING ####

# Get the top genre for all songs
myLabels = pd.read_csv(tracksDir, low_memory=False) # load data
labelMat = myLabels.values # turn into np matrix
topGenreIndx = np.where(labelMat[0,:] == 'genres') # find column with top genre info for songs
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

# Extract genre IDs as numbers from string
songGenreIds = extract_song_genre_ids(songGenres)

# Organize genres of songs into a binary feature vector
songGenreCode = genre_ids_to_binFeats(songGenreIds, allGenreIds, num_of_songs, num_of_genres)

########################################################
#### BREAK INTO TRAINING AND TESTING SETS ####

trainProp = 0.99 # proportion of data for training
num_of_training = round(num_of_songs*trainProp) # number of songs for training
trX = onlyFeats[0:num_of_training,:] # break data into training and testing sets
trY = songGenreCode[0:num_of_training,:]
teX = onlyFeats[(num_of_training+1):,:]
teY = songGenreCode[(num_of_training+1):,:]

########################################################
#### DESIGN MULTI LAYER PERCEPTRON MODEL ####

# Define functions for model
def init_weights(shape): # initializes random weights for perceptron
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, b_h, w_o, b_o):
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.nn.sigmoid(tf.matmul(h, w_o) + b_o) # note that we dont take the softmax at the end because our cost fn does that for us

# Define tensorflow variables
X = tf.placeholder("float", [None, num_of_features]) # batch_size by input data (batch size not yet determined)
Y = tf.placeholder("float", [None, num_of_genres]) # batch_size by output data (batch size not yet determined)

# Define neural network parameters
networksToTry = np.power(2, (np.arange(9.) + 1)) # num of neurons in input layer in each network
epochs = 100 # epochs to train networks
accMetric = np.zeros([len(networksToTry), epochs]) # create matrix to store network average accuracy
for k in range(0, len(networksToTry)):
    numOfNeurons = int(networksToTry[k]) # num of neurons in input layer
    w_h = init_weights([num_of_features, numOfNeurons]) # create symbolic variables
    b_h = init_weights([numOfNeurons])
    w_o = init_weights([numOfNeurons, num_of_genres])
    b_o = init_weights([num_of_genres])

    # Define model
    py_x = model(X, w_h, b_h, w_o, b_o)

    # Define cost function, training optimizer, and prediction operation
    # modified cross entropy to explicit mathematical formula of sigmoid cross entropy loss
    #cost = -tf.reduce_sum( (  (Y*tf.log(py_x + 1e-9)) + ((1-Y) * tf.log(1 - py_x + 1e-9)) )  , name='xentropy' )
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
    predict_op = py_x

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        batch_size = 1000 # training batch size
        num_of_batches = int(num_of_training/batch_size) # number of batches

        t0 = time.time() # start timer
        for i in range(epochs): # use this for loop to do batch training in epochs
            print('For network ', str(k+1), 'Starting epoch ', str(i + 1))
            ptr = 0
            for j in range(num_of_batches):
                inp, out = trX[ptr:ptr+batch_size], trY[ptr:ptr+batch_size]
                ptr+=batch_size
                sess.run(train_op, feed_dict={X: inp, Y: out}) # train network
                prY = sess.run(predict_op, feed_dict={X: teX}) # get predictions

            # Compute Jaccard index for estimate
            labelInt = np.sum(np.multiply(prY == 1, teY), 1) # intersection of predicted and true labels
            labelUni = np.sum(np.ceil(((prY == 1) + teY)/2), 1) # union of predicted and true labels
            jacIndx = np.divide(labelInt, labelUni) # Jaccard index of each prediction
            avgJacIndx = np.mean(jacIndx) # calculate average Jaccard index
            accMetric[k, i] = avgJacIndx # store average Jaccard index

        t1 = time.time() # end timer
        totalTime = t1 - t0 # get time for this iteration
        print('     Process took ', str(totalTime), ' seconds.')

    sess.close()

#################################################
#### SAVE DATA ####

# Save average accuracy metric for MLP results
np.savetxt('Results/accMetric_MLP.csv', accMetric, delimiter=",")

# Save the number of neurons in the input layer
np.savetxt('Results/numberOfNeurons_MLP.csv', networksToTry, delimiter=",")

#####################################################
#### PLOT RESULTS ####

maxAcc = np.amax(accMetric, axis=1)
plt.plot(networksToTry, maxAcc, 'k')
plt.xlabel('Number of neurons in input layer')
plt.ylabel('Average Jaccard Index')
plt.title('Perceptron network performance')
plt.savefig('Figures/PerceptronPerformance_topGenre.png')
plt.close()
