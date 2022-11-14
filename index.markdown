---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

Team: Vanessa Yip (kyip31), Kevin Wu (kwu333), Yi-Ting Chiang (ychiang48), Daniel Wu (dwu362), Chengrui Li (cli420)

# CS 7641 Machine Learning Group Project

## Introduction/Background
Music has been a huge part in our daily lives. In 2021 alone, the music industry market generated $61.82 billion. As producers, it would be invaluable to be able to predict the success of a song. With Spotify being the largest music streaming service provider possessing over 400 million active users, the goal of this project is to predict whether a song will land in Spotify’s Top 200 chart. Relevant researches have been done on analysis of song components, lyrics, music emotions and more, which are referenced below.  

We will focus on two datasets found on kaggle. The Spotify Chart dataset (https://www.kaggle.com/datasets/dhruvildave/spotify-charts) includes songs that are in the Top 200 Chart since 2017. The Spotify dataset (https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=tracks.csv) includes all songs on Spotify released from year 1921 to 2020. We will also extract further information about songs and artists through Spotify’s Web API. 

## Problem Definition
If Rihanna returns from hiatus and decides to release a new album after eight years, how likely are her songs going to trend on streaming platforms? Was anyone able to foresee Olivia Rodrigo’s success with her debut album? Combination of music components, including genre, artist attributes, song characteristics all contribute towards the success of a song, and machine learning algorithm could serve as an extremely useful method for producers and companies to predict the successes of their songs.  

Our team aims to construct a model that determines whether a newly released song will be included in Spotify’s Top 200 chart. Ultimately, we believe that this model could help drive music production processes and set up a song for success even before it is released. 

# Midterm Report #

## Data Collection
There is plenty of data collected on Spotify on the internet. Our data originated from two datasets on Kaggle: 1) [Spotify Top 200 Charts](https://www.kaggle.com/datasets/dhruvildave/spotify-charts) and 2) [All Songs on Spotify](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=dict_artists.json). We were able to join the two datasets by joining rows by linking the song ID in one table to the song URL in the other table (by parsing out the song ID in the URL).

There is plenty of data collected on Spotify on the internet. Our data originated from four sources:
Songs that were on the Top 200 chart from January 2017 to December 2021(https://www.kaggle.com/datasets/dhruvildave/spotify-charts)
1. Song attributes of 600K+ Spotify tracks released from 1900 to April 2021(https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=tracks.csv)
2. Spotify’s API (https://developer.spotify.com/documentation/web-api/reference/#/)
Artist attributes (https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=artists.csv)

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

### Principal Component Analysis
We perform Principal Component Analysis (PCA) via our own coded implementation on our main dataset to reduce the number of features to 2. Since our dataset is fairly large, with over 50000 observations and 16 different features, it is important to find ways to reduce the dimensionality of our data so that we can reduce the complexity of our analysis while still preserving the most important parts of our data. The results ran on our dataset are plotted in the graph below:
![PCA Results](/docs/assets/midterm_pca.png)

## Modeling
### Support Vector Machines - Supervised Learning
The first model we implemented to test on our datasets was SVM with linear, polynomial, RBF kernels. We chose to test this model first, as it is more computationally efficient and accurate on datasets with larger feature vectors. Indeed, the results followed this trend. As shown below, we received a mean accuracy of 0.859 across the three different kernels. Although this model produces fairly accurate results, we suspect there might be some issues of overfitting. For the final report, we aim to add cross validation and regularization (e.g. Lasso) of our dataset before running SVM and seeing if we can still achieve high accuracy amidst these conditions.

Model Building:
1. Keep the numerical features
2. Scale the data
3. Select our features using forward selection and backward elimination
4. The results for both methods are the same, and the selected features are as follow:
['acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'followers_total', 'instrumentalness', 'liveness', 'loudness', 'mode', 'popularity_total', 'valence']
5. Run the SVM Model + Logistic Regression Model
6. Compare the results: accuracy, confusion matrix

*Results - Confusion Matrices*

Linear Kernel:

![Linear SVM Confusion Matrix](/docs/assets/linear_confusion.png)

Polynomial Kernel:

![Polynomial SVM Confusion Matrix](/docs/assets/poly_confusion.png)

RBF Kernel:

![RBF SVM Confusion Matrix](/docs/assets/rbf_confusion.png)


*Results - Accuracy*

| Kernel Type       | Accuracy |
|-------------------|-------|
| Linear      | 0.855 |
| Polynomial  | 0.862 |
| RBF         | 0.859 |


# Future Goals

### Supervised Methods  
- K-nearest neighbors as a starting point for classification of smaller samples of our dataset, but we anticipate scaling it will be impractical for the entire dataset.
- Gaussian naïve bayes, random forest, decision trees 

### Unsupervised Methods  
- Agglomerative clustering, k-means  
- Neural networks (eg: CNN)

Our initial data exploration will establish baseline models and later transition to higher-performance models. We anticipate that non-linear models such as boosting trees (XGBoost) or neural networks will outperform linear models. 

## References
[1] Pham, J., Kyauk, E., & Park, E. (2016). Predicting song popularity. Dept. Comput. Sci., Stanford Univ., Stanford, CA, USA, Tech. Rep, 26. 

[2] J. S. Gulmatico, J. A. B. Susa, M. A. F. Malbog, A. Acoba, M. D. Nipas and J. N. Mindoro, "SpotiPred: A Machine Learning Approach Prediction of Spotify Music Popularity by Audio Features," 2022 Second International Conference on Power, Control and Computing Technologies (ICPC2T), 2022, pp. 1-5, doi: 10.1109/ICPC2T53885.2022.9776765. 

[3] Al-Beitawi, Z., Salehan, M., & Zhang, S. (2020). “What makes a song trend? Cluster analysis of musical attributes for Spotify top trending songs”. Journal of Marketing Development and Competitiveness, 14(3), 79-91. 

[4] Martín-Gutiérrez, D., Peñaloza, G. H., Belmonte-Hernández, A., & García, F. Á. (2020). “A multimodal end-to-end deep learning architecture for music popularity prediction”. IEEE Access, 8, 39361-39374. 

## Proposed Timeline and Member Responsibilities

![Gantt Chart](/docs/assets/gantt_chart.png)

## Contributions Table
Vanessa:
- Data collection, data preprocessing, introduction, Problem Definition

Kevin:
- Dataset and reference papers, methods of analysis

Yi-Ting:
- Dataset and reference papers, proposal video

Daniel:
- Data preprocessing, model evaluations, website, potential results and discussion, proposal video

Chengrui:
- Gantt chart, dataset
