# DOCS

## Table of Contents

[[toc]]


## BRIEF VIEW

This dataset contains videos from 4 different YouTubers and all the comments made on those
videos. The primary objective of this dataset is to cluster the comments to identify a cluster that
contains all the spam comments and fix the issue once and for all.

[DATASET](https://www.kaggle.com/datasets/japkeeratsingh/youtube-videos-and-the-comments)


## MACHINE LEARNING PIPELINE

A Machine Learning pipeline is a process of automating the workflow of a complete 
machine learning task. It can be done by enabling a sequence of data to be transformed 
and correlated together in a model that can be analyzed to get the output. A typical pipeline 
includes raw data input, features, outputs, model parameters, ML models, and Predictions. 
Moreover, an ML Pipeline contains multiple sequential steps that perform everything ranging 
from data extraction and pre-processing to model training and deployment in Machine 
learning in a modular approach. It means that in the pipeline, each step is designed as an 
independent module, and all these modules are tied together to get the final result. 



## ABOUT THE DATASET

The dataset consists of youtube comments from four different youtubers. It consists of their several 
videos and our aim is to detect the comment which is classified as spam. 
YouTubers in this dataset include -
1. Cleo Abram 
2. Physics Girl 
3. Jet Lag: The Game 
4. Neo 


Primary objective of this dataset is to cluster the comments to identify a cluster that contains all the spam 
comments and fix the issue once and for all.

## DATA FETCH
Since the dataset is of large size we mount it in drive and then upload it in colab using google drive 
and then perform sampling to reduce size.

Importing libraries

```python
import pandas as pd import matplotlib.pyplot as plt 
import seaborn as sns import numpy as np 
from sklearn.metrics import silhouette_score
```

```python
from google.colab import drive 
drive.mount('/content/drive')
```
Loading dataset(importing)

```python
df1= pd.read_csv("/content/drive/MyDrive YT_Videos_Comments.csv") 
df1
```

the shape of dataset is 861962 rows * 9 columns 
Now separating the dataset of four youtubers and then sampling(example – For Cleo Abram) 

```python
df2=df1.iloc[0:28000] df2 import random 
df=df2.sample(frac=0.125) 
```

frac: float, optional
Fraction of axis items to return. (here 0.125) 

## DATA PREPROCESSING AND FEATURE SELECTION

A real-world data generally contains noises, missing values, and maybe in an unusable format which 
cannot be directly used for machine learning models. Data preprocessing is required tasks for 
cleaning the data and making it suitable for a machine learning model which also increases the 
accuracy and efficiency of a machine learning model.
Checking number of null valued using info and then dropping the null rows

```python
df.info() df.dropna()
```

Feature selection means selecting the essential features for further pipeline tasks from the original 
dataset and dropping the rest 
Since the sole purpose of the project is to detect spam comments we will be selecting only the 
comments displayed as a feature and leave the rest.


## VECTORIZATION

Vectorization is the process of converting textual data into numerical vectors and is a process that is 
usually applied once the text is cleaned. It can help improve the execution speed and reduce the 
training time of your code. 
A generic natural language processing (NLP) model is a combination of multiple mathematical and 
statistical steps. It usually starts with raw text and ends with a model that can predict outcomes. In 
between, there are multiple steps that include text cleaning, modeling, and hyperparameter tuning. 
Text cleaning, or normalization, is one of the most important steps of any NLP task. It includes 
removing unwanted data, converting words to their base forms (stemming or lemmatization), and 
vectorization. 
There are three major methods for performing vectorization on text data: 
1. CountVectorizer 
2. TF-IDF 
3. Word2Vec


## TEXT PREPROCESSING

Whenever we have textual data, we need to apply several pre-processing steps to the data to 
transform words into numerical features that work with machine learning algorithms We will be 
using the NLTK (Natural Language Toolkit) library here.



```python
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download("stopwords")

```




Using the required libraries we perform following steps:-
1)Text lowercase 
2)Remove Numbers 
3)Remove punctuation 
4)remove whitespaces 
5)Remove default stopwords 

```python
stop_words = set(stopwords.words('english'))
```

6)Stemming: Stemming is the process of getting the root form of a word. Stem or root is the part to 
which inflectional affixes (-ed, -ize, -de, -s, etc.) are added.

7)Lemmatization: Like stemming, lemmatization also converts a word to its root form. The only 
difference is that lemmatization ensures that the root word belongs to the language. We will get 
valid words if we use lemmatization.

```python
lemmatizer = WordNetLemmatizer()
for comment in df['Comment (Actual)']: # Remove 
unwanted characters comment = re.sub(r"http\S+", "",
str(comment)) comment = re.sub('[^a-zA-Z0-9]+', ' ',
str(comment))
 
 # Tokenize comment tokens = 
nltk.word_tokenize(comment)
 
 # Remove stop words and lemmatize tokens 
preprocessed_tokens = [] for token in
tokens: if token.lower() not in
stop_words:
 preprocessed_tokens.append(lemmatizer.lemmatize(token.lower
()))
 
 # Join preprocessed tokens back into a single string 
preprocessed_comment = ' '.join(preprocessed_tokens) 
preprocessed_comments.append(preprocessed_comment)

```

## TFIDF VECTORIZER

TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. This is very common 
algorithm to transform text into a meaningful representation of numbers which is used to fit machine 
algorithm for prediction 
Count Vectorizer give number of frequency with respect to index of vocabulary where as tfidf
consider overall documents of weight of words.

Vectorization in python

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_comments) 
```

After vectorization we will be getting new set of features in the form of floating type values which is 
stored in x .we will now be using this for our further procedures.

## CLUSTERING (BEFORE PCA)

Clustering or cluster analysis is a machine learning technique, which groups the unlabelled dataset. It 
can be defined as "A way of grouping the data points into different clusters, consisting of similar 
data points. The objects with the possible similarities remain in a group that has less or no 
similarities with another group."
It does it by finding some similar patterns in the unlabelled dataset such as shape, size, color, 
behavior, etc., and divides them as per the presence and absence of those similar patterns. 
Here we use mainly popular Clustering algorithms that are widely used in machine learning: 
1)k-means algorithm 
2)DBSCAN algorithm 
3)Agglomerative Hierarchical algorithm

## SILHOUTTE SCORE

Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering 
technique. Its value ranges from -1 to 1.

Silhoutte Score=(b-a)/max(a,b) , Where 
a=average intra-cluster distance , i.e, the distance between each point within a cluster 
b= average inter-cluster distance, i.e , the average distance between all clusters

## KMEANS CLUSTERING

The k-means algorithm is one of the most popular clustering algorithms. It classifies the dataset by 
dividing the samples into different clusters of equal variances. The number of clusters must be 
specified in this algorithm. It is fast with fewer computations required, with the linear complexity of 
O(n).

```python
from sklearn.cluster import KMeans kmeans = KMeans(2)
kmeans.fit(X) labels_kmeans = kmeans.fit_predict(X)
```
The predicted values of k means clustering are stored in labels_kmeans 

## DBSCAN

It stands for Density-Based Spatial Clustering of Applications with Noise. It is an 
example of a density-based model similar to the mean-shift, but with some remarkable advantages. 
In this algorithm, the areas of high density are separated by the areas of low density. Because of this, 
the clusters can be found in any arbitrary shape.

```python
from sklearn.cluster import DBSCAN db_default 
= DBSCAN(eps=1.0).fit(X) labels_dbscan = 
db_default.fit_predict(X)
print(labels_dbscan)
```
Here we will use DBSCAN for detecting outliers

## HIERARCHICAL CLUSTERING

The Agglomerative hierarchical algorithm performs the bottom-up hierarchical clustering. In this, 
each data point is treated as a single cluster at the outset and then successively merged. The cluster 
hierarchy can be represented as a tree-structure.

```python
from sklearn.cluster import AgglomerativeClustering X=X.toarray()
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', l 
inkage='ward') 
labels_agglo=cluster.fit_predict(X)

```
## PLOTTING AND SCORES

For plotting clusters calculating the average TFIDF scores for each comment (taking sum of each 
word’s tfidf score and dividing by total number of features in X) and then plotting it with comment 
number.

```python
l=[] num=[] for i in range(0,len(X)):
 num.append(i)
 l.append(np.sum(X[i])/len(X[i])) print(l) 
plt.scatter(num,l, c = labels_kmeans, cmap= 
"Paired") plt.show()
```

Then calculating the silhouette scores corresponding to each clustering

```python
from sklearn.metrics import silhouette_score 
print(silhouette_score(X,labels_dbscan)) 
```
The silhouette scores obtained from above came out to be very less and also the clusters formed 
were not good enough. 
So for this we need to reduce the number of features , as the number of features obtained above is
equal to the number of unique words. In order to take every feature into account we perform 
dimensionality reduction using PCA(Principal Component analysis)

## PCA

Principal Component Analysis is an unsupervised learning algorithm that is used for the 
dimensionality reduction in machine learning. It is a statistical process that converts the observations 
of correlated features into a set of linearly uncorrelated features with the help of orthogonal 
transformation. These new transformed features are called the Principal Components. 
PCA generally tries to find the lower-dimensional surface to project the high-dimensional data. 
The PCA algorithm is based on some mathematical concepts such as:

1)Variance and Covariance
2)Eigenvalues and Eigen factors

```python
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X1 = pca.fit_transform(X)
print(X1)
```

Taking number of dimensions to be 2 and storing the new dataset in X1

## CLUSTERING AFTER PCA

Now we perform clustering on new dataset X1 obtained after performing PCA Kmeans

```python
from sklearn.cluster import KMeans 
kmeans1 = KMeans(2) kmeans1.fit(X1)
labels_kmeans1 = kmeans1.fit_predict(X1)
print(labels_kmeans1)
```

plotting scatterplot

```python
plt.scatter(X1.transpose()[0],X1.transpose()[1], c = labels_kmeans1, cm 
ap= "Paired") plt.show()
```
Finding best parameter values for DBSCAN which are eps and min_samples.

```python
min_samples = range(10,21) eps = np.arange(0.05,0.13, 0.01) # returns 
array of ranging from 0.05 t o 0.13 with step of 0.01 output = []
for ms in
min_samples: for
ep in eps:
 labels = DBSCAN(min_samples=ms, eps = ep).fit(X1).labels_ 
score = silhouette_score(X1, labels) output.append((ms,
ep, score))
min_samples, eps, score = sorted(output, key=lambda x:x[-1])[-1]
print(f"Best silhouette_score: {score}") print(f"min_samples: 
{min_samples}") print(f"eps: {eps}")
```

The above values are found corresponding to best silhouette score 
Similarly doing so for DBSCAN and Hierarchical clustering. Here we will perform prediction using 
kmeans and hierarchical only and use DBSCAN for detecting outliers only.

## DECISION BOUNDARY FOR K MEANS

Plotting decision boundary for clusters corresponding to kmeans clustering
```python
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier from sklearn.svm 
import SVC from sklearn import tree from sklearn.inspection import
DecisionBoundaryDisplay display=DecisionBoundaryDisplay.from_estimator( 
kmeans1, X1, alpha=0.4, plot_method='contourf' ,response_method="pr 
edict" ) display.ax_.scatter(X1[:, 0], X1[:, 1], c=labels_kmeans1,
s=20, edgecol or="k") display.ax_.set_title("kmeans")
plt.show()
```

## SOME VISUALISATIONS

For visualising the clusters better we will pot scatterplot using express 

```python
import plotly.express as px fig = 
px.scatter(x=a, y=b, color=labels_kmeans1)
fig.update_layout(
 title="kmeans clustering", 
xaxis_title="pc1", yaxis_title="pc2",
) fig.show()
```
Further, based on predicted values of kmeans and agglomerative clustering we will calculate the 
number of Spam comments and plot them in a pie chart Finding unique values and their number

```python
unique_kmeans1, counts_kmeans1 = np.unique(labels_kmeans1, return_count 
s=True) print(unique_kmeans1) print(counts_kmeans1) 
x=['NOT SPAM',
'SPAM']
```

plotting pie chart

```python
plt.pie(counts_kmeans1, labels = x,wedgeprops = { 'linewidth' : 1, 'edg 
ecolor' : 'white' },shadow=True,autopct='%1.2f%%') plt.show()
```
we can use both kmeans and hierarchical clustering as the final model as the silhouette scores of 
both are close to each other . 
The silhouette score of kmeans came out to be slightly more than agglomerative.
<!-- ### Routes

[contact](/contact)

[contact](/contact.md)

[contact](/dp.png)
(/dp.png)

### Code Blocks

```js
console.log('hello world')
```

### Tables

| Heading  | Column 2     | Column 3     |
| :------: | -----------: | ------------ |
| centered | right align  | left align   |
| zebra    | stripes      | too          | -->