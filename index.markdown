---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

Team: Vanessa Yip (kyip31), Kevin Wu (kwu333), Yi-Ting Chiang (ychiang48), Daniel Wu (dwu362), Chengrui Li (cli420)

# CS 7641 Machine Learning Group Project

## Introduction/Background
We will focus on two datasets found on kaggle. The Spotify Chart dataset (https://www.kaggle.com/datasets/dhruvildave/spotify-charts) includes songs that are in the Top 200 Chart since 2017. The Spotify dataset (https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=tracks.csv) includes all songs on Spotify released from year 1921 to 2020. We will also extract further information about songs and artists through Spotify’s Web API. 

We will focus on two datasets found on kaggle. The Spotify Chart dataset (https://www.kaggle.com/datasets/dhruvildave/spotify-charts) includes songs that are in the Top 200 Chart since 2017. The Spotify dataset (https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?resource=download&select=tracks.csv) includes all songs on Spotify released from year 1921 to 2020. We will also extract further information about songs and artists through Spotify’s Web API. 

## Problem Definition
If Rihanna returns from hiatus and decides to release a new album after eight years, how likely are her songs going to trend on streaming platforms? Was anyone able to foresee Olivia Rodrigo’s success with her debut album? Combination of music components, including genre, artist attributes, song characteristics all contribute towards the success of a song, and machine learning algorithm could serve as an extremely useful method for producers and companies to predict the successes of their songs.  

Our team aims to construct a model that determines whether a newly released song will be included in Spotify’s Top 200 chart. Ultimately, we believe that this model could help drive music production processes and set up a song for success even before it is released. 

## Methods
### Data Preprocessing  
- Initially perform PCA to reduce dimensionality of dataset and extract the most relevant features that maximize variance in the data. Depending on how much of the available Spotify we deem as relevant, we will also perform incremental  
- Normalized data and split into training and testing sets (8:2) for cross validation. The validation set will be a subset of the training set. 

### Supervised Methods  
- K-nearest neighbors as a starting point for classification of smaller samples of our dataset, but we anticipate scaling it will be impractical for the entire dataset.  
- Support vector machine has been one of the most ubiquitous classifiers for music classification. As SVMs are a binary classifier, we will utilize a radial basis function (RBF) kernel to enlarge the feature space.  
- Gaussian naïve bayes, random forest, decision tree 

### Unsupervised Methods  
- Agglomerative clustering, k-means  
- Neural networks (this might a bit too ambitious) 

Our initial data exploration will establish baseline models and later transition to higher-performance models. We anticipate that non-linear models such as boosting trees (XGBoost) or neural networks will outperform linear models. 

## Potential Results and Discussion
There will definitely be some quantitative measurement of accuracy that will be used in our metrics. We have a few options: 

One is simply the accuracy score. We want to measure how accurate we are in predicting whether a list of songs ends up in the top 200 trending or not. 

Another viable option that might produce smoother results would be using R^2 score. Instead of just a binary label, we can attribute song’s with an actual ranking from 1-200 or even to beyond 200. This would give us a better measure of how close our predictions were to their actual rankings. 

Alternative metrics we will explore include NPV, specificity, and AUC. 

## References
[1] James Pham, Edric Kyauk, and Edwin Park. “Predicting Song Popularity”. http://cs229.stanford.edu/proj2015/140_report.pdf 

[2] J. S. Gulmatico, J. A. B. Susa, M. A. F. Malbog, A. Acoba, M. D. Nipas and J. N. Mindoro, "SpotiPred: A Machine Learning Approach Prediction of Spotify Music Popularity by Audio Features," 2022 Second International Conference on Power, Control and Computing Technologies (ICPC2T), 2022, pp. 1-5, doi: 10.1109/ICPC2T53885.2022.9776765. 

[3] Zayd Al-Beitawi, Mohammad Salehan and Sonya Zhang. “What Makes a Song Trend? Cluster Analysis of Musical Attributes for Spotify Top Trending Songs”, 2020. http://www.na-businesspress.com/JMDC/JMDC14-3/8_Al-BeitawiFinal.pdf 

## Proposed Timeline and Member Responsibilities

![Gantt Chart](/docs/assets/gantt_chart.png)

## Contributions Table
Vanessa:
- Introduction, Problem Definition
Kevin:
- Dataset and reference papers, methods of analysis
Yi-Ting:
- Dataset and reference papers, proposal video
Daniel:
- website, potential results and discussion, proposal video
Chengrui:
- Gantt chart, dataset
