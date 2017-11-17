# Exploring the Free Music Archive (FMA) dataset
Using tensorflow to classify music genre and popularity of songs in the Free Music Archive. 

## Motivation
The amount of music available increased significantly after the democratization of music production. We would think that this process would allow for smaller and less-reknowned artists to find an audience and make a living from their art. However, we must remember that when given [too many choices](https://en.wikipedia.org/wiki/The_Paradox_of_Choice), people usually get stressed and stick to what they already know. Therefore, the democratization of music production and the increase in the volume of music produced may have the opposite effect on the budding music performer. Moreover, if the search algorithms which parse through music data bias their classification and categorization of music, then it may keep the consumer from discovering music they would really like and be willing to pay for. This mismatch in value and cost can lead to instances of market failure. 

Companies like Pandora, Spotify, Apple and others manage large quantities of music data which need to be classified and categorized to make them easily accessible. Thankfully, most of the music produced comes with human-given labels like the artist, the genre, and even the production company. However, to make music recommendation fair to smaller or less reknowned artists and more useful to the consumer looking for music, we must see if the music itself (audio signal) can provide the necessary features for classification and categorization. 

## Data
The data comes from the [Free Music Archive (FMA)](https://freemusicarchive.org/) which is an interactive library of high-quality, legal downloads. Specifically, it was downloaded from the resources provided by the [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma/) github page. I downloaded the [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) file for which I got the link from the above mentioned github page. Please check out the [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma/) github page, if you need more details about the data.  

#### Data description
I performed analysis on the following aspects of the data. 

1. The feature set ---- 106,574 tracks with 518 features extracted for each track. 
2. The genre labels ---- Each track is labeled with one or more genres, for a total 163 genres.
3. The number of listens ---- Each track is labeled with a integer number of listens, providing data for the popularity of the track.

## Data exploration
Initially, I wanted to predict the genre of a song given its audio features so I started by visualizing the distribution of genres in the data set. Below are the top ten genres in the dataset. Keep in mind that one song can have more than one genre. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/topTenGenres.png "Top ten genres")

And the following are the bottom ten genres. 
(For the code used for visualization please refer to the [Code](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Code) folder in the repo. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/botTenGenres.png "Bottom ten genres")

It seems that some genres occur far more often than others in this dataset. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/GenreOccurence.png "Genre occurrence")

The large majority of genres occur less than 2,000 times in the dataset while there are two genres that occur more than 20,000 times. Training a predictor on data like this will be difficult since there will be a lot of class imbalance. Additionally, this classification problem has more than two classes (genres) for all data and more than two labels (genres) for each song making it a multi-class, multi-label classification task. 

## Genre classification with k-nearest neighbor
The beyond-human interpretable number of dimensions in the feature set makes it unclear if the feature space can accurately predict the genre of a song over a random guess. Therefore, I started with a simple k-nearest neighbor (k-NN) approach which is non-parametric and can reveal if the features-space has some mapping to the label-space. This was implemented with the help of the tensorflow library to speed up processing time (found in the [Code](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Code) folder).

For k-NN runs with more than 1 neighbor, I used a simple "voting" mechanism with majority rule. Neighbors (training data) vote which genre labels to apply to the new point (testing data). Genres with 50% or more of the votes are predicted as labels for the test point. I used the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) as the accuracy metric. The Jaccard index is commonly used to measure accuracy for multi-label problems and is calculated by dividing the number of elements in intersection of the predicted and actual labels by the number of elements in the union of the predicted and actual labels. The Jaccard index breaks down in the following way:

1. If the multi-label prediction is absolutely correct, then the intersection and union of the predicted and actual labels are the same number and the Jaccard Index = 1.
2. If the multi-label prediction is partially correct, the the intersection is smaller than the union of the predicted and actual labels. In this case 0 < Jaccard Index < 1.
3. The multi-label prediction is totally wrong and the intersection is 0 making the Jaccard index = 0. 

The following is a visualization for the mean and standard deviation of the average Jaccard indices for mulitple runs of k-NN for which the training and testing sets were randomized. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/kNNGenrePredictor.png "Jaccard index of k-NN runs")

The Jaccard index never reaches 1 even for best case. Moreover, it's strange that a k-NN with only 1 nearest neighbor performed the best. Below is the distribution of Jaccard indices for one run of the k-NN where k=1 was used to test about 1000 testing points. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/JaccIndxHistforK_1.png "Jaccard index distribution for one run of k-NN with k=1")

The k-NN method of genre prediction gets at least one of the labels right about 40% of the time. However, it gets all of the labels wrong about 60% of the time. This is still far better than random guessing considering there are 163 labels but maybe this performance can be increased. 

## Reducing the feature and label space 

Repo under construction. 
