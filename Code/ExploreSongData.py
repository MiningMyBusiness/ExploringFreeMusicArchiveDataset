# import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Define location of relevant files
tracksDir = 'fma_metadata/tracks.csv' # metadata for every track
genreDir = 'fma_metadata/genres.csv' # genre key from ID to string

#######################################
#### SONG GENRE (LABEL) PROCESSING ####

# Get the genre for all songs
myLabels = pd.read_csv(tracksDir,  low_memory=False) # load data
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
songGenreIds = list() # create list to populate with genre IDs
for genres in songGenres: # for each string in the list
    commaIndx = [m.start() for m in re.finditer(',', genres)] # find the position of commas
    if len(genres) == 2: # genre only has two brackets (no genre)
        songGenreIds.append([0]) # add 0 for genre Id
    if len(commaIndx) == 0 and len(genres) > 2: # no commas in the string and only one genre
        songGenreIds.append([int(genres[1:-1])]) # append the genre to the list
    if len(commaIndx) > 0 and len(genres) > 2: # if a song has more than one genre
        tempList = list() # create temporary list
        for x in range(0,len(commaIndx) + 1): # go through each comma index and append genres
            if x == 0:
                oneGenre = int(genres[1:commaIndx[x]])
            elif x > 0 and x < len(commaIndx):
                oneGenre = int(genres[commaIndx[x-1]+1:commaIndx[x]])
            elif x == len(commaIndx):
                oneGenre = int(genres[commaIndx[x-1]+1:-1])
            tempList.append(oneGenre) # append each genre to the temporary list
        songGenreIds.append(tempList) # append temporary list to full list

# Organize genres of songs into a binary feature vector
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

# Visualize how many times a genre appears in the data set
num_of_songs_byGenre = np.sum(songGenreCode, 0) # add all instances of a genre across the dataset
plt.hist(num_of_songs_byGenre, 50)
plt.xlabel('Number of times genre appears in dataset')
plt.ylabel('Number of genres')
plt.title('Frequency of Genre Occurrence')
plt.savefig('Figures/GenreOccurence.png')
plt.clf()
plt.close()

# Visualize most occuring genres
sortIndex = np.argsort(num_of_songs_byGenre)
topTenGenres = allGenreNameWithNoGenre[sortIndex[-10:]]
barIndex = np.arange(10)
topTenNums = num_of_songs_byGenre[sortIndex[-10:]]
plt.bar(barIndex, topTenNums)
plt.xticks(barIndex, topTenGenres)
plt.xticks(rotation=65)
plt.ylabel('Number of times genre appears in dataset')
plt.rc('xtick',labelsize=8)
plt.gcf().subplots_adjust(bottom=0.3)
plt.title('Top Ten Occurring Genres')
plt.savefig('Figures/topTenGenres.png')
plt.clf()
plt.close()

# Visualize least occuring genres
botTenGenres = allGenreNameWithNoGenre[sortIndex[:10]]
barIndex = np.arange(10)
botTenNums = num_of_songs_byGenre[sortIndex[:10]]
plt.bar(barIndex, botTenNums)
plt.xticks(barIndex, botTenGenres)
plt.xticks(rotation=65)
plt.ylabel('Number of times genre appears in dataset')
plt.rc('xtick',labelsize=8)
plt.gcf().subplots_adjust(bottom=0.3)
plt.title('Bottom Ten Occurring Genres')
plt.savefig('Figures/botTenGenres.png')
plt.clf()
plt.close()

###############################################################
#### SONG LISTENS PROCESSING ####

# Get the number of listens for all songs
topListenIndx = np.where(labelMat[0,:] == 'listens') # find column number of listens
songListens = labelMat[2:,topListenIndx[0][1]] # get song listens
songListens = songListens.astype(np.float)
