# Code overview
This folder contains all of the code files used to do the analysis presented in the README of the master directory. 

## Code descriptions 
Please find below short descriptions of what each code file does. 

#### ExploreSongData.py
This code visualizes aspects of the fma_metadata data set for song genres and listens.

#### GenrePredictor_DeepNeuralNet.py
Implements a deep neural net in tensorflow to train and predict the genre labels for each song using the features of the audio signal.

#### GenrePredictor_MultiLayerPerceptron.py
Implements a multilayer perceptron (input layer and output layer) in tensorflow to train and predict genre labels for each song. 

#### GenrePredictor_kNN_labelAndFeatReduction.py
Implements a k-nearest neighbors classifier in tensorflow with dimensionality reduction of the feature-space (with PCA) and the label-space (with LDA).

#### GenreTopicModeling.py
Explores topic modeling of genre labels with LSI and LDA for dimensional reduction of label-space. 

#### PredictGenre_NearestNeighbor.py
Implements a k-nearest neighbors classifier in tensorflow with full feature and label space. 

#### PredictGenre_NearestNeighbor_featDimReduce.py
Implements a k-nearest neighbors classifier in tensorflow with dimensionally reduced feature space (with PCA) but full label space. 
