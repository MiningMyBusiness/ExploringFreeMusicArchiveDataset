## Title: Genre topic modeling with LSI and LDA
##  This code attempts to reduce the genre space with topic modeling using latent
##  semantic indexing and latent dirichlet allocation.
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
import re
import csv

#############################################
#### FUNCTION/MODULES ####

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

####################################################
#### FILE LOCATIONS ####

# Define location of relevant files
tracksDir = 'fma_metadata/tracks.csv' # metadata for every track
genreDir = 'fma_metadata/genres.csv' # genre key from ID to string

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

# Create Latent Semantic Indexing (LSI) model with as many topics as genres
myLsiModel = models.lsimodel.LsiModel(corpus, id2word=dictionary, num_topics=num_of_genres)

# Save the first 10 topics to file
myLsiTopics = myLsiModel.print_topics(10)
np.savetxt('Results/myLsiTopics_topten.txt', myLsiTopics)

# Do variance analysis of LSI model
myVarExpl = myLsiModel.projection.s # get variance explained per topic
totVar = np.sum(myVarExpl) # get total variance
sumVarPerTopic = np.zeros([myVarExpl.shape[0], 1]) # create variable to store incremental variance explained with each additional topic
for i in range(0, myVarExpl.shape[0]): # for each topic
    sumVarPerTopic[i] = np.sum(myVarExpl[0:i])/totVar # add variance explained to that topic

# Plot LSI results and dimensionality of dataset
topicNumber = np.arange(myVarExpl.shape[0]) + 1 # create array of number of topics
plt.plot(topicNumber, sumVarPerTopic)
plt.plot(topicNumber, [0.8]*topicNumber.shape[0], 'r')
plt.xlabel('Number of topics (genre groups) in song genres')
plt.ylabel('Proportion of variance explained in songe genres')
plt.title('Latent Semantic Analysis of Song Genres')
plt.savefig('Figures/LSASongGernes_VarExpl.png')
plt.clf()
plt.close()

# Create LDA model with corpus and with fewer topics
myLdaModel = models.ldamodel.LdaModel(corpus, num_topics=70, id2word = dictionary, passes=10)

# Save a random 10 topics to file from the LDA model
myLdaTopics = myLdaModel.print_topics(10)
np.savetxt('Results/myLdaTopics_topten.txt', myLdaTopics)
