# import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from keras.layers import Input, Dense, Dropout
from keras.models import Model
import keras
from keras import backend as K

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Load in and process data
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define location of relevant files
featureDir = 'fma_metadata/features.csv' # features for each track
tracksDir = 'fma_metadata/tracks.csv' # metadata for every track
genreDir = 'fma_metadata/genres.csv' # genre key from ID to string

# define function to get numpy array of song features
# this keeps lots of variables with redundant data out of RAM
def getFeatures(featureDir):
    myFeatures = pd.read_csv(featureDir, low_memory=False) # load data
    featureMat = myFeatures.values # turn to np matrix
    onlyFeats = featureMat[3:,1:].astype(float) # extract just features w/o feature labels
    return onlyFeats

# define function to normalize features
def normalize_features(featureMat):
    dimMax = featureMat.max(0)
    dimMin = featureMat.min(0)
    dimRange = dimMax - dimMin
    normMat = featureMat - np.tile(dimMin, [featureMat.shape[0], 1]) # subtract mean from matrix
    normMat = np.divide(normMat, np.tile(dimRange, [featureMat.shape[0], 1])) # divide by standard deviation
    return normMat, dimMax, dimMin

# Get features of songs
print('Grabbing song features...')
onlyFeats = getFeatures(featureDir)
onlyFeats, dimMax, dimMin = normalize_features(onlyFeats)
num_of_features = onlyFeats.shape[1]

# define function to get numpy array of song genre
def getGenre(tracksDir):
    myLabels = pd.read_csv(tracksDir, low_memory=False) # load data
    labelMat = myLabels.values # turn into np matrix
    topGenreIndx = np.where(labelMat[0,:] == 'genres_all') # find column with top genre info for songs
    songGenres = labelMat[2:,topGenreIndx[0][0]] # get song genres
    return songGenres

# Get the top genre for all songs
print('Grabbing song genres...')
songGenres = getGenre(tracksDir)
num_of_songs = songGenres.shape[0]

# define function to grab genre names
def getGenreNames(genreDir):
    genreNames = pd.read_csv(genreDir) # load in file with genre info
    genreMat = genreNames.values # extract values from the data frame
    allGenreName = genreMat[:,3] # get the names of genres
    allGenreIds = genreMat[:,0] # get all genre ids
    genreToIndx = {}
    idToGenre = {}
    indxToGenre = {}
    counter = 0
    for name,id in zip(allGenreName, allGenreIds):
        genreToIndx[name] = counter
        indxToGenre[counter] = name
        idToGenre[id] = name
        counter += 1
    return genreToIndx, idToGenre, indxToGenre

# Get number of genres and genre names
print('Creating dictionaries for genre name and ID...')
genreToIndx, idToGenre, indxToGenre = getGenreNames(genreDir)
num_of_genres = len(genreToIndx.keys()) + 1 # the total number of genres and plus 1 if a song has no genre listed

# create matrix of labels with genre IDs
# define function to create list of song genres
def extract_song_genre_names(genres,  num_of_genres, idToGenre, genreToIndx):
    thisSongLabels = np.zeros(num_of_genres)
    commaIndx = [m.start() for m in re.finditer(',', genres)] # find the position of commas
    if len(genres) == 2: # genre only has two brackets (no genre)
        thisSongLabels[-1] = 1
    if len(commaIndx) == 0 and len(genres) > 2: # no commas in the string and only one genre
        genreId = int(genres[1:-1]) # get genre ID as an integer
        genreName = idToGenre[genreId] # pull genre name from array
        thisSongLabels[genreToIndx[genreName]] = 1
    if len(commaIndx) > 0 and len(genres) > 2: # if a song has more than one genre (at least one comma in the string)
        for x in range(0,len(commaIndx) + 1): # go through each comma index and append genres ignoring starting and ending brackets
            if x == 0:
                oneGenreId = int(genres[1:commaIndx[x]])
            elif x > 0 and x < len(commaIndx):
                oneGenreId = int(genres[commaIndx[x-1]+1:commaIndx[x]])
            elif x == len(commaIndx):
                oneGenreId = int(genres[commaIndx[x-1]+1:-1])
            oneGenreName = idToGenre[oneGenreId]
            thisSongLabels[genreToIndx[oneGenreName]] = 1
    return thisSongLabels

# go through each song and grab the genres
print('Create label matrix of song genres...')
genreLabels = np.zeros((num_of_songs, num_of_genres))
for i,genres in enumerate(songGenres):
    thisSongLabels = extract_song_genre_names(genres,  num_of_genres, idToGenre, genreToIndx)
    genreLabels[i,:] = thisSongLabels

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build deep autoencoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Building autoencoder model...')

# encoded dimension
encoded_dim = 164

# number of classes
num_classes = num_of_genres

# input placeholder
input_img = Input(shape=(num_of_features,))
# 'encoder' model
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(encoded_dim, activation='relu')(encoded)

# 'decoder' model
decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(num_of_features, activation='sigmoid')(decoded)

# create seperate encoder
encoder = Model(input_img, encoded)

# full autoeconder model
autoencoder = Model(input_img, decoded)

# configure model to be trained
# per-feature binary crossentropy loss
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the autencoder model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Training autoencoder model...')

# shuffle array and labels
rng_state = np.random.get_state()
np.random.shuffle(onlyFeats)
np.random.set_state(rng_state)
np.random.shuffle(genreLabels)
propTest = 0.2
nTrain = int((1-propTest)*num_of_songs)

autoencoder.fit(onlyFeats[0:nTrain,:], onlyFeats[0:nTrain,:],
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(onlyFeats[nTrain:,:], onlyFeats[nTrain:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Validate the info compression of autencoder model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Validating autoencoder model...')

from sklearn.metrics import r2_score
y_pred = autoencoder.predict(onlyFeats[nTrain:,:])
varExpl = r2_score(onlyFeats[nTrain:,:], y_pred,
            multioutput='variance_weighted')
print('Proportion of variance explained: ' + str(varExpl))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build classifier network with encoder on the bottom
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Building classifier model...')

x = encoder.output
# add a classification layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_of_genres, activation='sigmoid')(x)

# this is the model we will train for classification
model = Model(encoder.input, predictions)

# first we need to freeze all encoder layers so they
# are not trained by the classification process
for layer in encoder.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the classifier model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Training classifier model...')

model.fit(onlyFeats[0:nTrain,:], genreLabels[0:nTrain,:],
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(onlyFeats[nTrain:,:], genreLabels[nTrain:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Validate the classifier model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Validating classifier model...')

from sklearn.metrics import jaccard_similarity_score
label_pred = model.predict(onlyFeats[nTrain:,:])
threshs = np.arange(0, 0.5, 0.01)
jacIndxs = np.zeros(threshs.shape)
for i,thresh in enumerate(threshs):
    thisJacIndx = jaccard_similarity_score(genreLabels[nTrain:,:], label_pred > thresh)
    jacIndxs[i] = thisJacIndx

plt.plot(threshs, jacIndxs)
plt.xlabel('Thresholds')
plt.ylabel('Jaccard index')
plt.title('Jaccard index from thresholds of DNN output')
plt.savefig('Figures/DnnOuputThreshJaccIndx.jpg')
plt.close()

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Threshold predictions and calculate accuracy for each genre
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Calculating accuracy, precision, and recall...')

# calculate accuracy, precision, and recall
optThresh = threshs[np.argmax(jacIndxs)]
label_pred = label_pred > optThresh
isRight = np.logical_not(np.logical_xor(label_pred, genreLabels[nTrain:,:]))
genreAcc = np.sum(isRight, axis=0)/isRight.shape[0]
genrePres = np.sum(np.logical_and(label_pred, genreLabels[nTrain:,:]), axis=0)/np.sum(label_pred, axis=0)
genreRecall = np.sum(np.logical_and(label_pred, genreLabels[nTrain:,:]), axis=0)/np.sum(genreLabels[nTrain:,:], axis=0)

# plot accuracy
plt.plot(genreAcc)
plt.xlabel('Genre index')
plt.ylabel('Accuracy')
plt.title('Accuracy by genre')
plt.savefig('Figures/AccuracyGenre.jpg')
plt.close()

# get genres with less than 90% accuracy
accLess90 = np.where(genreAcc < 0.9)
less90List = []
less90Acc = []
for index in accLess90[0]:
    thisGenre = indxToGenre[index]
    less90List.append(thisGenre)
    less90Acc.append(genreAcc[index])
plt.bar(less90List, less90Acc)
plt.xlabel('Genre')
plt.ylabel('Accuracy')
plt.xticks(rotation=20)
plt.title('Genres with less than 90% accuracy')
plt.savefig('Figures/AccLessThan90.jpg')
plt.close()

# plot accuracy vs precision
plt.scatter(genreAcc, genrePres)
plt.xlabel('Accuracy')
plt.ylabel('Precision')
plt.title('Accuracy vs. Precision for genres')
plt.savefig('Figures/AccVsPres.jpg')
plt.close()

# plot accuracy vs recall
plt.scatter(genreAcc, genreRecall)
plt.xlabel('Accuracy')
plt.ylabel('Recall')
plt.title('Accuracy vs. Recall for genres')
plt.savefig('Figures/AccVsRecall.jpg')
plt.close()
