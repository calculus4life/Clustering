#!/usr/bin/env python
# coding: utf-8

# Description
# 
# Context
# AllLife Bank wants to focus on its credit card customer base in the next financial year. They have been advised by their marketing research team, that the penetration in the market can be improved. Based on this input, the Marketing team proposes to run personalized campaigns to target new customers as well as upsell to existing customers. Another insight from the market research was that the customers perceive the support services of the back poorly. Based on this, the Operations team wants to upgrade the service delivery model, to ensure that customer queries are resolved faster. Head of Marketing and Head of Delivery both decide to reach out to the Data Science team for help

# Objective
# To identify different segments in the existing customer, based on their spending patterns as well as past interaction with the bank, using clustering algorithms, and provide recommendations to the bank on how to better market to and service these customers.

# Data Description
# 
# 
# The data provided is of various customers of a bank and their financial attributes like credit limit, the total number of credit cards the customer has, and different channels through which customers have contacted the bank for any queries (including visiting the bank, online and through a call center).

# Data Dictionary
# 
# 
# Sl_No: Primary key of the records
# 
# Customer Key: Customer identification number
# 
# Average Credit Limit: Average credit limit of each customer for all credit cards
# 
# Total credit cards: Total number of credit cards possessed by the customer
# 
# Total visits bank: Total number of visits that customer made (yearly) personally to the bank
# 
# Total visits online: Total number of visits or online logins made by the customer (yearly)
# 
# Total calls made: Total number of calls made by the customer to the bank or its customer service department (yearly)

# In[1]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# # Reading the Dataset

# In[2]:


# loading the dataset
data = pd.read_excel("Credit+Card+Customer+Data.xlsx")


# # Checking the shape of the dataset

# In[3]:


data.shape


# # Displaying few rows of the dataset

# In[4]:


# viewing a random sample of the dataset

data.sample(5)


# # Creating a copy of original data

# In[5]:


# copying the data to another variable to avoid any changes to original data

df = data.copy()


# In[6]:


# dropping the serial no. column as it does not provide any information

df.drop("Sl_No", axis = 1, inplace=True)


# # Checking the data types of the columns for the dataset

# In[7]:


df.info()


# # Statistical summary of the dataset

# In[8]:


df.describe()


# The column "Avg_Credit_Limit", The average credit limit is around 34,574.24, with a standard deviation of 37,625.49. The credit limits range from a minimum of 3,000 to a maximum of 200,000.
# 
# For the column "Total_Credit_Cards,"  On average, customers have approximately 4.71 credit cards, with a standard deviation of 2.17. The number of credit cards ranges from a minimum of 1 to a maximum of 10.
# 
# In the "Total_visits_bank" column, On average, customers visit the bank approximately 2.40 times, with a standard deviation of 1.63. The number of visits to the bank ranges from a minimum of 0 to a maximum of 5.
# 
# In the "Total_visits_online" column, On average, customers visit online platforms approximately 2.61 times, with a standard deviation of 2.94. The number of online visits ranges from a minimum of 0 to a maximum of 15.
# 
# Regarding the "Total_calls_made" column, there are 660 non-null values. On average, customers make approximately 3.58 calls, with a standard deviation of 2.87. The number of calls made ranges from a minimum of 0 to a maximum of 10.

# # Checking for missing values

# In[9]:


# checking for missing values

df.isnull().sum()


# - There are no missing values in our data

# In[10]:


df['Avg_Credit_Limit'].unique()


# In[11]:


df['Total_Credit_Cards'].unique()


# In[12]:


df['Total_visits_bank'].unique()


# In[13]:


df['Total_visits_online'].unique()


# In[14]:


df['Total_calls_made'].unique()


# # Data Visualization

# In[15]:


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[16]:


histogram_boxplot(data = df, feature = 'Avg_Credit_Limit')


# In[17]:


histogram_boxplot(data = df, feature = 'Total_Credit_Cards')


# In[18]:


histogram_boxplot(data = df, feature = 'Total_visits_bank')


# In[19]:


histogram_boxplot(data = df, feature = 'Total_visits_online')


# In[20]:


histogram_boxplot(data = df, feature = 'Total_calls_made')


# # Identifying correlation

# In[21]:


# select the numerical features as a list of numerical columns
num = df.select_dtypes(include= np.number).columns.tolist()
num


# In[22]:


# Let's check for correlations
num_data = data[num]
cor = num_data.corr()
cor


# In[23]:


num_data


# In[24]:


plt.figure(figsize = (12, 7))
sns.heatmap(cor, annot = True, vmin = -1, vmax = 1, cmap = 'Spectral')
plt.title('Heatmap showing correlation of feactures with target');


# The heatmap plot uncovers significant correlations within the dataset, shedding light on the interplay between different variables. These correlations not only indicate whether the relationships are positive or negative but also provide insights into the strength of these connections. By examining the heatmap, we can discern intriguing patterns and gain a deeper understanding of how variables interact and influence one another.

# # Outlier Detection and Handling

# In[25]:


#To get the outlier using the boxplot

plt.figure(figsize = (20, 30))

for i, variable in enumerate(df):
    plt.subplot(5,4, 1 + i)
    plt.boxplot(df[variable])
    plt.title(variable)
plt.show()


# In[26]:


def treat_outliers(num_data, col):
    '''treat the outliers in variable
    col: str, name of the numerical variable
    video_model: dataframe'''
    Q1 = df[col].quantile(0.25) #25th quantile
    Q3 = df[col].quantile(0.75) #75th quantile
    IQR = Q3 - Q1
    lower_whisker = Q1 - (1.5 * IQR)
    upper_whisker = Q3 + (1.5 * IQR)
    num_data[col] = np.clip((df[col]), lower_whisker,upper_whisker)
    return df


def treat_outliers_all(num_data, col_list):
    for c in col_list:
        num_data = treat_outliers(num_data, c)
    return num_data


# In[27]:


num_data = treat_outliers_all(df, num_data)


# In[28]:


#To confirm and check this outliers has been treated

#To get the outlier using the boxplot

plt.figure(figsize = (20, 30))

for i, variable in enumerate(num_data):
    plt.subplot(5,4, 1 + i)
    plt.boxplot(df[variable])
    plt.title(variable)
plt.show()


# # K-Means Clustering 

# In[29]:


# scaling the dataset before clustering
scaler = StandardScaler()
num_data_sd = scaler.fit_transform(num_data)


# In[30]:


kmeans = KMeans(random_state=0)

kmeans.fit(num_data_sd)


# In[31]:


kmeans.inertia_


# In[32]:


y_pred = kmeans.predict(num_data_sd)

y_pred


# In[33]:


silhouette_score(num_data_sd, y_pred)


# # Choosing optimal number of clusters

# In[34]:


clusters = range(1, 9)
meanDistortions = []  # Create a empty list

for k in clusters:
    model = KMeans(n_clusters=k)  # Initialize KMeans
    model.fit(num_data)  # Fit kMeans on the data
    prediction = model.predict(num_data_sd)  # Predict the model on the data
    distortion = (
        sum(np.min(cdist(num_data, model.cluster_centers_, "euclidean"), axis=1))
        / num_data.shape[0]  # Find distortion
    )

    meanDistortions.append(
        distortion
    )  # Append distortion values to the empty list created above

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")  # Title of X-axis
plt.ylabel("Average Distortion")  # Title of y-axis
plt.title("Selecting k with the Elbow Method", fontsize=20)  # Title of the plot


# In[35]:


sil_score = []  # Create empty list
cluster_list = list(range(2, 10))  # Creating a list of range from 2 to 10
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters)  # Initializing KMeans algorithm
    preds = clusterer.fit_predict((num_data_sd))  # Predicting on the data
    # centers = clusterer.cluster_centers_
    score = silhouette_score(num_data_sd, preds)  # Cacalculating silhouette score
    sil_score.append(score)  # Appending silhouette score to empty list created above
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

plt.plot(cluster_list, sil_score)


# # Applying KMeans clustering for k=3

# In[36]:


kmeans3 = KMeans(n_clusters=3, random_state=0)

kmeans3.fit(num_data_sd)


# In[37]:


kmeans3.labels_


# In[38]:


# adding kmeans cluster labels to the original dataframe

num_data["Kmeans_clusters"] = kmeans3.labels_


# In[39]:


fig, axes = plt.subplots(1, 4, figsize=(16, 6))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(4):
    sns.boxplot(ax=axes[ii], y=num_data[num[counter]], x=num_data["Kmeans_clusters"])
    counter = counter + 1

fig.tight_layout(pad=2.0)


# In[40]:


df.groupby("K_means_segments").mean().plot.bar(figsize=(15, 6))


# # DBSCAN Clustering

# In[42]:


from sklearn.cluster import DBSCAN


# In[43]:


# create instance of DBSCAN
dbscan = DBSCAN()

# fit and predict the labels
db_labels = dbscan.fit_predict(num_data_sd)


# In[44]:


# check the labels

np.unique(db_labels)


# In[45]:


#try values of eps btn 0.1 and 1; minPts btn 2 and 10
eps = np.linspace(0.1, 1, 10)

minPts = np.arange(2, 10)


# In[49]:


import itertools


# create a tuple of eps and MinPts using product method of itertools

hyper_list= list(itertools.product(eps, minPts))


# # Silhouette score

# In[50]:


for eps, minPts in hyper_list:
    dbscan = DBSCAN(eps=eps, min_samples=minPts)
    labels = dbscan.fit_predict(num_data_sd)
    unique_labels = np.unique(labels)
if len(unique_labels) > 1:
    score = silhouette_score(num_data_sd, labels)
    print(f"eps: {eps}; minPts: {minPts}; num_labels: {len(unique_labels)}; score: {score}")


# # Rebuild the DBSCAN model

# In[51]:


# create instance of DBSCAN
dbscan1 = DBSCAN(eps = 1.0, min_samples = 3)

# fit and predict the labels
db_labels1 = dbscan1.fit_predict(num_data_sd)


# In[52]:


# Check the unique values
np.unique(db_labels1)


# In[53]:


# add the dbscan labels to the original data

data["db_labels"] = db_labels1


# In[54]:


fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Boxplot of numerical variables for dbscan")
counter = 0
for ii in range(5):
    sns.boxplot(ax=axes[ii], y=data[num[counter]], x=data["db_labels"])
    counter = counter + 1
fig.tight_layout(pad=2.0)


# In[55]:


data.query("db_labels == -1")


# In[56]:


data.query('db_labels == 1')


# In[57]:


data.query('db_labels == 0')


# In[ ]:




