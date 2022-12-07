---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

Team: Kwan Wing Yip (kyip31), Kevin Wu (kwu333), Yi-Ting Chiang (ychiang48), Daniel Wu (dwu362), Chengrui Li (cli420)

# CS 7641 Machine Learning Group Project

## Introduction/Background
Music has been a huge part in our daily lives. In 2021 alone, the music industry market generated $61.82 billion. As producers, it would be invaluable to be able to predict the success of a song. With Spotify being the largest music streaming service provider possessing over 400 million active users, the goal of this project is to predict whether a song will land in Spotify’s Top 200 chart. Relevant researches have been done on analysis of song components, lyrics, music emotions and more, which are referenced below.  

## Problem Definition
If Rihanna returns from hiatus and decides to release a new album after eight years, how likely are her songs going to trend on streaming platforms? Was anyone able to foresee Olivia Rodrigo’s success with her debut album? Combination of music components, including genre, artist attributes, song characteristics all contribute towards the success of a song, and machine learning algorithm could serve as an extremely useful method for producers and companies to predict the successes of their songs.  

Our team aims to construct a model that determines whether a newly released song will be included in Spotify’s Top 200 chart. Ultimately, we believe that this model could help drive music production processes and set up a song for success even before it is released. 

## Data Collection

There is plenty of data collected on Spotify on the internet. Our data originated from three sources:
1. [Top 200](https://www.kaggle.com/datasets/dhruvildave/spotify-charts): Songs that were on the chart from January 2017 to December 2021
2. [Song attributes of 600K+]((https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=tracks.csv)): Spotify tracks released from 1900 to April 2021
3. [Spotify’s API](https://developer.spotify.com/documentation/web-api/reference/#/)
4. [Artist attributes](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=artists.csv)

## Data Visualization and Preprocessing

### Correlation Matrix
We generated a correlation matrix to help us understand our dataset better. Here, we can see which features are positively or negatively correlated to each other, and to what extent. In the future, this will allow for more fine-tuned feature selection via backward feature elimination that will produce more versatile and robust results in our testing on different models. 

![Correlation Matrix](/docs/assets/correlation_matrix.png)

### Joining all the datasets
The first dataset allowed us to identify songs that are in the top 200 chart and the second dataset contains attributes of each song (e.g. tempo, liveness, acousticness). By joining the first two datasets, we acquired a dataset that contains attributes of each song as well as a column labeling whether the song was a top 200 song. As the first dataset has song ID and the date the song was charted as the primary key, one song may appear multiple times. Hence we dropped duplicates and only took unique songs out of the first dataset. We also matched the timeframe of the two datasets and limited it to January 2017 to April 2021 to avoid biased comparison. 

As we were preprocessing the data, we noticed that some songs that were listed in the top 200 chart were missing in the second dataset. As data in the second dataset was originally extracted from Spotify’s API, we extracted song and artist data from Spotify APIs via known unique song identifiers. 

Afterwards, we joined the above dataset with the third dataset which contains attributes about each artists (e.g. number of followers on Spotify, popularity). As each song may be sung more than one artist, with a maximum of 58 artists per song, we decided to first look at the distribution of the number of artists per song, with the following results:

![Artists per Song](/docs/assets/artists_per_song.png)

With further investigation, we also extracted the information below regarding number of artists for a song at different percentiles:
- 80%: 1
- 90%: 2
- 97.5%: 3

Upon such analysis, we decided to truncate the number of artists per song to 3 and joined the dataset with the artists dataset. After which we added two new columns, ‘followers_total’ and ‘popularity_total’ which sums up the followers and popularity of the 3 (or fewer) artist(s). 

### Downsampling
Additionally, we discovered a class imbalance between songs that were and were not labeled as top 200.

![Class imbalance](/docs/assets/class_imbalance.png)

A song that was in the top 200 is labeled 0 and vice versa. As seen above, there is a great imbalance between the two classes. We tackled this problem by downsampling the songs that were not in the top 200 so that the number of songs in each were roughly equal. 


### Principal Component Analysis
We perform Principal Component Analysis (PCA) via our own coded implementation on our main dataset to reduce the number of features to 2. Since our dataset is fairly large, with over 50000 observations and 16 different features, it is important to find ways to reduce the dimensionality of our data so that we can reduce the complexity of our analysis while still preserving the most important parts of our data. The results ran on our dataset are plotted in the graph below:
![PCA Results](/docs/assets/midterm_pca.png)

### Outlier Rejection
Additionally, we computed z-scores for all quantitative features in our dataset to detect any outliers. Boxplots for each of the features are displayed below.

![Outliers](/docs/assets/outliers.png)

An experimentally determined threshold z-score of ±5.199 was applied to determine the amount of outliers within the dataset with respect to each individual feature. A summary of the outlier counts is as shown below. While all the song data points identified through this method could have been excluded, we opted to examine each of these outlier features more closely and compare them to the feature descriptions as defined by Spotify API documentation. For some of the features where outliers were detected, namely “loudness” and “speechiness”, the data has already been scaled by Spotify’s algorithms. Moreover, the feature “time_signature” was a discrete categorical variable (0-5) and thus outlier removal was not necessary. For these reasons, these features were not considered when removing outliers in our data. Finally, we did remove 84 songs due to “duration_ms” so that the song duration feature could be scaled appropriately when used to train and test our models.

Here are the amount of outliers detected for each feature:

![Amount of Outliers](/docs/assets/amt_outliers.png)

## Modeling

Model Building:
1. Keep the numerical features
2. Scale the data
3. Select our features using forward selection and backward elimination
4. The results for both methods are the same, and the selected features are as follow:
['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'followers_total', 'instrumentalness', 'liveness', 'loudness', 'mode', 'popularity_total', 'valence']
5. Run the SVM Model + Logistic Regression Model
6. Compare the results: accuracy, confusion matrix

### KMeans Clustering - Unsupervised Learning
One of the simplest unsupervised learning models is the KMeans clustering algorithm. This algorithm initializes k centroids at random at then iteratively subdivides the dataset into “k” clusters around the initialized centroids. We ran our data on several values of k, ranging from 2 to 20, producing comparable results. We ran the KMeans Clustering algorithm on our datasets for values of k ranging from 2 to 9, achieving accuracies in the range 0.8 to 0.9.
![KMeans Results](/docs/assets/kmeans_results.png)

### Gaussian Mixture Model - Unsupervised Learning
A better performing unsupervised learning model is the Gaussian Mixture Model (GMM), since it provides more flexibility in clustering by outputting the probability of a datapoint belonging to a cluster rather than a hard clustering like KMeans. We used cross-validation to split our dataset into training and testing data, running 10 iterations where 9 batches were used for training and the last batch was used for testing. To our surprise, the GMM did not perform as well as our KMeans Clustering algorithm, performing at an accuracy of only 0.688. 

### DBScan - Unsupervised Learning
The DBSCAN algorithm takes in two parameters, the minimum samples to be considered a cluster and the minimum distance between points. In order to test the different parameters, we set “eps”  between values (1e-4 - 10) and the “min_samples” between (2 - 500). Even after trying all of these, the DBSCAN algorithm still has trouble clustering a majority of the values presumably because many of these values are extremely noisy (pun intended). Our best accuracy came out to be 50.4% which is marginally better than the expected value of random guess. 

### Support Vector Machines - Supervised Learning
The first supervised model we implemented to test on our datasets was SVM with linear, polynomial, RBF kernels. We chose to test this model first, as it is more computationally efficient and accurate on datasets with larger feature vectors. Indeed, the results followed this trend. As shown below, we received a mean accuracy of 0.877 across the three different kernels. Although this model produces fairly accurate results, we suspect there might be some issues of overfitting. For the final report, we aim to add cross validation and regularization (e.g. Lasso) of our dataset before running SVM and seeing if we can still achieve high accuracy amidst these conditions.

*Results - Confusion Matrices*

Linear Kernel:

![Linear SVM Confusion Matrix](/docs/assets/linear_confusion.png)

Polynomial Kernel:

![Polynomial SVM Confusion Matrix](/docs/assets/poly_confusion.png)

RBF Kernel:

![RBF SVM Confusion Matrix](/docs/assets/rbf_confusion.png)


*Results - Accuracy*

Since our dataset labels are a simple boolean, we use accuracy classification score to evaluate the performance of each model. Accuracy is simply defined as the number of labels predicted correctly divided by the total number of data observations.

| Kernel Type       | Accuracy |
|-------------------|-------|
| Linear      | 0.879 |
| Polynomial  | 0.874 |
| RBF         | 0.879 |

### Logistic Regression - Supervised Learning
The next model we tested on our dataset was logistic regression, another supervised method. This was a no-brainer, as our dataset labels are binary values, so this fits the needs of the logistic regression model perfectly. This model was fairly simple to implement, and it searches for a line to separate the two labels in our dataset. Again, we ended up with relatively high accuracies for this model, at about 0.8885.

*Results - Confusion Matrix*

![Logistic Confusion Matrix](/docs/assets/logistic_confusion.png)

*Results*

Accuracy: 0.85

Misclassification Rate: 0.15

True Positive Rate: 0.8830

True Negative Rate: 0.8187

Precision: 0.8222 

### Random Forest - Supervised Learning
For the Random Forest Classifier, a model was made with 1000 decision trees and the minimum samples in a leaf did not go below 5% of the original dataset. Our accuracy was 85.6%. The results are shown below:

![Random Forest Result](/docs/assets/random_forest_res.png)

Here is an example of one tree in the forest:

![Random Forest Example](/docs/assets/random_forest_example.png)

### Neural Network - Supervised Learning
The Neural Network was made with two hidden layers. Some hyperparameter tuning was applied to the Neural Network to discover which parameters worked better for our dataset. Upon inspection, the top 15 accuracy results did not include “sigmoid” and “softmax” functions.  
These are the parameters:

![NN params](/docs/assets/nn_params.png)

These are the top 15 results: 

![NN results](/docs/assets/nn_results.png)

## Conclusion
Our models in the end generally performed at higher levels of accuracy than we had expected. Although some of our unsupervised models struggled with forming clusters amongst our dataset, all of our supervised models were able to hit accuracies over 80%. One main reason for this is most likely due to the large dimensionality of our dataset, making it harder for the unsupervised models to form clusters that correctly labeled our data.

With these trained models and their decently accurate predictions on whether or not a song will end up in the Top 200 Charts, we hope up and coming artists can utilize our tool to analyze certain attributes of new songs that may increase its chances of being more popular.

## References
[1] Pham, J., Kyauk, E., & Park, E. (2016). Predicting song popularity. Dept. Comput. Sci., Stanford Univ., Stanford, CA, USA, Tech. Rep, 26. 

[2] J. S. Gulmatico, J. A. B. Susa, M. A. F. Malbog, A. Acoba, M. D. Nipas and J. N. Mindoro, "SpotiPred: A Machine Learning Approach Prediction of Spotify Music Popularity by Audio Features," 2022 Second International Conference on Power, Control and Computing Technologies (ICPC2T), 2022, pp. 1-5, doi: 10.1109/ICPC2T53885.2022.9776765. 

[3] Al-Beitawi, Z., Salehan, M., & Zhang, S. (2020). “What makes a song trend? Cluster analysis of musical attributes for Spotify top trending songs”. Journal of Marketing Development and Competitiveness, 14(3), 79-91. 

[4] Martín-Gutiérrez, D., Peñaloza, G. H., Belmonte-Hernández, A., & García, F. Á. (2020). “A multimodal end-to-end deep learning architecture for music popularity prediction”. IEEE Access, 8, 39361-39374. 

## Proposed Timeline and Member Responsibilities

![Gantt Chart](/docs/assets/gantt_chart.png)

## Contributions Table
Kwan Wing:
- Data collection, data preprocessing, introduction, Problem Definition

Kevin:
- API data collection, data preprocessing, dataset and reference papers, methods of analysis

Yi-Ting:
- Correlation Heatmap, variables selection, model building (Logistic regression), dataset and reference papers, proposal video

Daniel:
- Data preprocessing, model evaluations, website, potential results and discussion, proposal video

Chengrui:
- Data preprocessing, SVM (‘linear’,’polynomial’,’rbf’) models, Gantt chart, dataset
