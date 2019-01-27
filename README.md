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

## Music genre prediction 
Initially, I wanted to predict the genre of a song given its audio features so I started by visualizing the distribution of genres in the data set. Below are the top ten genres in the dataset. Keep in mind that one song can have more than one genre. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/topTenGenres.png "Top ten genres")

And the following are the bottom ten genres. 
(For the code used for visualization please refer to the [Code](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Code) folder in the repo. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/botTenGenres.png "Bottom ten genres")

It seems that some genres occur far more often than others in this dataset. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/GenreOccurence.png "Genre occurrence")

The large majority of genres occur less than 2,000 times in the dataset while there are two genres that occur more than 20,000 times. Training a predictor on data like this will be difficult since there will be a lot of class imbalance. Additionally, this classification problem has more than two classes (genres) for all data and more than two labels (genres) for each song making it a multi-class, multi-label classification task. 

### Genre classification with k-nearest neighbor
The beyond-human interpretable number of dimensions in the feature set makes it unclear if the feature space can accurately predict the genre of a song over a random guess. Therefore, I started with a simple k-nearest neighbor (k-NN) approach which is non-parametric and can reveal if the features-space has some mapping to the label-space. This was implemented with the help of the tensorflow library to speed up processing time (found in the [Code](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Code) folder).

For k-NN runs with more than 1 neighbor, I used a simple "voting" mechanism with majority rule. Neighbors (training data) vote which genre labels to apply to the new point (testing data). Genres with 50% or more of the votes are predicted as labels for the test point. I used the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) as the accuracy metric. The Jaccard index is commonly used to measure accuracy for multi-label problems and is calculated by dividing the number of elements in intersection of the predicted and actual labels by the number of elements in the union of the predicted and actual labels. The Jaccard index breaks down in the following way:

1. If the multi-label prediction is absolutely correct, then the intersection and union of the predicted and actual labels are the same number and the Jaccard Index = 1.
2. If the multi-label prediction is partially correct, the the intersection is smaller than the union of the predicted and actual labels. In this case 0 < Jaccard Index < 1.
3. The multi-label prediction is totally wrong and the intersection is 0 making the Jaccard index = 0. 

The following is a visualization for the mean and standard deviation of the average Jaccard indices for mulitple runs of k-NN for which the training and testing sets were randomized. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/kNNGenrePredictor.png "Jaccard index of k-NN runs")

The average Jaccard index for a k-NN run never reaches 1 even for best case. Moreover, it's strange that a k-NN with only 1 nearest neighbor performed the best. Below is the distribution of Jaccard indices for one run of the k-NN where k=1 was used to test about 1000 testing points. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/JaccIndxHistforK_1.png "Jaccard index distribution for one run of k-NN with k=1")

The k-NN method of genre prediction gets at least one of the labels right about 40% of the time. However, it gets all of the labels wrong about 60% of the time. This is still far better than random guessing considering there are 163 labels but maybe this performance can be increased. I also tried a multilayer perceptron with up to 518 input layer neurons, and a deep neural net with one hidden layer with up to 518 hidden neurons (both implemented with tensorflow) and a Random Forest classifier which performed far worse than k-NN. The Python scripts are available in the [Code](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Code) folder of the repo.  

### Reducing the feature-space 
Before I attempted to improve the performance of the genre classification, I wanted to reduce the dimensionality of the feature space down from 518 dimensions to improve processing time. I performed [principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) to find that about 155 principal components explained some 90% of the variance in the dataset of features. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/PCASongFeatures.png "PCA analysis of audio features")

This suggested that I would lose minimal information in the feature dataset and still be able to reduce the dimensionality almost five-fold. I performed the dimensionality reduction shrinking the original feature set of 106,574 by 518 to 106,574 by 155. I did k-NN classification of genres in this reduced feature-space to make sure that the transformation and information loss did not significantly change the mapping from the feature-space to the label-space. 

k-NN with reduced dimensional feature-space. 
![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/kNNGenrePredictor_featDimReduce.png "k-NN with reduced dimensional features")

We can see that the Jaccard index didn't change appreciable between the two cases. This lack of change suggests that the dimensionality reduction of the feature-space did not influence the ability to correctly classify the genre.

### Topic modelling the genre-space
Reducing the dimensionality of the genre-space may help in increasing the Jaccard index. One approach to do so is by using [topic modeling](https://en.wikipedia.org/wiki/Topic_model) techniques since the genre space is binary (a song either belongs to a genre or not). The most common methods of topic modeling include [Latent Semantic Indexing (LSI)](https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing) or [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation). Both can be used to group words (or genres) into topics (or genre groups) based how often they occur with each other. 

However, each method has major differences. LSI is akin to PCA and is faster than LDA. LDA is Bayesian update method that must go through the data many times (number of passes) to produce results. While the results genrated by LDA are more human-interpretable than those generated by LSI, LSI allows for a visualization of the dimensionality of the binary label-space. I performed LSI on the genre labels to see how many genre groups were needed to capture most of the variance the genre-space. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/LSASongGernes_VarExpl.png "Latent semantic analysis (indexing) of song genres")

There is no clear "elbow" in the plot above suggesting that there may not be any inherent dimensionality in the genre-space from this dataset. This result is unsurprising since about 80 of the genres appear less than 1,000 times in a dataset with over 100,000 samples. It does seem that 70 topics from the LSI model can capture about 80% of the information in the genre space. 

However the topics in LSI are difficult to interpret. Below are the first four topics extracted by LSI. They are listed in order of "importance" in the dataset. 

1. '0.628*"Experimental" + 0.496*"Electronic" + 0.375*"Rock" + '
  '0.208*"Instrumental" + 0.144*"Avant-Garde" + 0.143*"Pop" + 0.130*"Noise" + '
  '0.119*"Ambient" + 0.106*"Folk" + 0.104*"Electroacoustic"'),
  
2. '-0.766*"Rock" + 0.276*"Electronic" + 0.269*"Experimental" + -0.262*"Punk" + '
  '-0.184*"Pop" + -0.158*"Lo-Fi" + -0.147*"Indie-Rock" + 0.132*"Instrumental" '
  '+ -0.118*"Folk" + -0.099*"Garage"'),

3. '0.699*"Electronic" + -0.546*"Experimental" + -0.236*"Avant-Garde" + '
  '-0.128*"Electroacoustic" + 0.120*"Ambient Electronic" + -0.109*"Improv" + '
  '-0.108*"Noise" + 0.107*"Instrumental" + 0.106*"Hip-Hop" + 0.103*"Pop"'),

4. '-0.684*"Instrumental" + -0.357*"Ambient" + -0.322*"Soundtrack" + '
  '-0.257*"Folk" + 0.254*"Electronic" + -0.221*"Pop" + 0.148*"Experimental" + '
  '-0.143*"Experimental Pop" + -0.092*"Singer-Songwriter" + 0.092*"Punk"')

Notice that a single genre appears in more than one topic. Sometimes this makes sense. For instance, "Rock" appears in the first topic with a positive weigth and in the second topic with a negative weight. However, this seems mostly to differentiate between Experimental and Electronic music that can and cannot be considered Rock, since both of these genres have positive ratings in the first two topics. 

To make the results more interpretable I also performed LDA with a preset of 70 topics. These are 5 random topics from LDA. Notice that all weights on genres are positive making the composition of these topics easier to understand. 

1. '0.548*"Free-Folk" + 0.422*"Folk" + 0.029*"Experimental" + 0.000*"Big '
  'Band/Swing" + 0.000*"Lounge" + 0.000*"South Indian Traditional" + '
  '0.000*"Thrash" + 0.000*"Black-Metal" + 0.000*"Drum & Bass" + 0.000*"Goth"'),

2. '0.564*"Avant-Garde" + 0.304*"Experimental" + 0.053*"New Wave" + '
  '0.043*"Progressive" + 0.035*"Rock" + 0.000*"Black-Metal" + 0.000*"Big '
  'Band/Swing" + 0.000*"Lounge" + 0.000*"Goth" + 0.000*"Drum & Bass"'),

3. '0.609*"Drone" + 0.391*"Experimental" + 0.000*"20th Century Classical" + '
  '0.000*"Drum & Bass" + 0.000*"Lounge" + 0.000*"Goth" + 0.000*"Black-Metal" + '
  '0.000*"Big Band/Swing" + 0.000*"South Indian Traditional" + '
  '0.000*"Shoegaze"'),

4. '0.818*"Blues" + 0.162*"Rock" + 0.019*"Gospel" + 0.000*"Big Band/Swing" + '
  '0.000*"Shoegaze" + 0.000*"Rap" + 0.000*"Drum & Bass" + 0.000*"Lounge" + '
  '0.000*"Goth" + 0.000*"Black-Metal"'),

5. '0.490*"Ambient Electronic" + 0.359*"Electronic" + 0.151*"Experimental" + '
  '0.000*"South Indian Traditional" + 0.000*"Rap" + 0.000*"Drum & Bass" + '
  '0.000*"Lounge" + 0.000*"Goth" + 0.000*"Black-Metal" + 0.000*"Big '
  'Band/Swing"')
  
Since these topics are much more interpretable and LDA categorizes songs into fewer topics, I compressed the genre-space from 164 genres to 70 topics learned with LDA. Then I attempted k-NN prediction of topics for each song with the reduced dimensional feature space. To recap, the genre-space has been reduced from 164 to 70 and the feature space has been reduced from 518 to 155. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/kNNGenrePredictor_featAndLabelDimReduce.png "k-NN with reduced dimensional feature-space and genre-space")

With this large reduction in the dimensionality of the problem, k-NN performs slightly worse at predicting the topics of the songs. This dataset has some genres that appear far more often than others. Even with topic modeling some of these genres are not captured as topics. It is hard to train any classifier on the less common genres since there is little data. 

But let's try a deep neural net (DNN) to classify the genre since DNNs can perform dimensionality reduction, manifold projections, and classification all at once. 

### Deep neural net classifier 
A deep neural net can perform non-linear dimensionality reduction of the feature space by encoding it in a latent dimensional space and use the encoding to classify the genre of the song. There are two ways to approach this problem. 

1) We can build one deep neural net that takes the song features as input and directly outputs the genres of the song as the prediction. This neural net would have a loss function defined by the genres of the songs. It would be trained on the training set and validated on how well it predicted the genre of the testing set. 

2) The other approach is to first build a deep encoder-decoder (autoencoder) network that can learn to represent the song features in a lower dimensional space and then reconstruct the songs from that space. The encoder-decoder net would have a loss function defined by the song features istelf because it's trying to reconstruct them. The encoder-decoder network would be trained on training set features and be validated on it's ability to reconstruct the test set features. Then we can take the encoder portion of the network and add layers to build a classifing neural net that uses the output of the encoder to predict the song genres which would have the genre labels of the song as the loss function. This network would be trained on the training set to predict genres and validated on the genres of the testing set. 

Let's take Approach 1 first since it's simpler. 

![alt text](https://github.com/MiningMyBusiness/ExploringFreeMusicArchiveDataset/raw/master/Figures/DnnOutputThresJaccIndx_justClassifier.jpg "Jaccard index from thresholding output of the classifying neural network.")
