# Playstore_Dataset_Analysis

<!DOCTYPE html>
<html lang="en">

<body>

<h1>Google Playstore Data Analytics</h1>

<h2>Introduction</h2>
<p>
    The Play Store apps data has enormous potential to drive app-making businesses to success. This dataset, chosen from Kaggle, contains 600k Play Store apps. We use this dataset to predict preferred positions as we have sufficient columns in this dataset to derive numerous insights. The dataset consists of 10 million rows and 24 columns. We ensure that all machine learning algorithms studied are implemented accordingly to the data.
</p>

<h2>Objective</h2>
<p>
    The main objective of this project is to predict the rating of one app from 600k apps, specifically the app named “ENGLISH”. We predict the rating based on app features such as whether the app is free, the category it belongs to, and other attributes, aiming for the most accurate prediction possible.
</p>

<h2>Insights</h2>
<ul>
    <li>Total Apps name present on Google Playstore.</li>
    <li>Count all categories according to genre.</li>
    <li>Count and display all entertainment categories.</li>
    <li>Count and display all Music & Audio categories.</li>
    <li>Count and display all Education categories.</li>
    <li>Count and display all Book & Reference categories.</li>
    <li>Count and display all Personalization categories.</li>
    <li>Count the values of all Ratings in ascending order.</li>
    <li>Count the values of Rating Count.</li>
    <li>Count the values of Installs.</li>
    <li>Count the free apps in the dataset.</li>
    <li>Count the top 50 free apps.</li>
    <li>Show the values of how many apps are released on the same date.</li>
    <li>Show the values of how many apps were last updated on the same date.</li>
    <li>Show the content rating according to the genre.</li>
    <li>Show the top 20 content ratings according to the genre.</li>
    <li>Count the value and show whether an app is ad-supported or not.</li>
    <li>Show how many apps offer in-app purchases.</li>
    <li>Show the value of editor's choice, whether it is true or false.</li>
    <li>Show the installs according to different categories.</li>
    <li>Show the top 30 apps by install value.</li>
    <li>Show all installs that are less than 100,000,000.</li>
    <li>Show the top 20 categories according to the installs.</li>
    <li>Show the top 10 installs according to the content rating.</li>
    <li>Show all the categories according to the count.</li>
    <li>Show all the categories according to the rating.</li>
</ul>

<h2>Importing Libraries</h2>
<pre><code>import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
</code></pre>

<h2>Uploading the Dataset (Getting the raw data)</h2>
<pre><code>df = pd.read_csv("Google-Playstore.csv")
</code></pre>

<h2>Descriptive Analysis</h2>
<p>
    <strong>Storing dataset into pandas DataFrame:</strong>
</p>
<pre><code>print("Dataset")
print("Rows and Columns", df.shape)
df
</code></pre>
<p>
    <strong>Show the top 10 values:</strong>
</p>
<pre><code>df.head(10)
</code></pre>
<p>
    <strong>Show the last 10 values:</strong>
</p>
<pre><code>df.tail(10)
</code></pre>
<p>
    <strong>Type of DataFrame:</strong>
</p>
<pre><code>type(df)
</code></pre>
<p>
    <strong>Show the location of row number 10472:</strong>
</p>
<pre><code>df.iloc[10472]
</code></pre>
<p>
    <strong>Slicing:</strong>
</p>
<pre><code>df.iloc[10469:10472]
</code></pre>
<p>
    <strong>Give all the info about all columns:</strong>
</p>
<pre><code>df.info()
</code></pre>
<p>
    <strong>Show all column names:</strong>
</p>
<pre><code>df.columns
</code></pre>

<h2>Cleaning of Data</h2>
<p>
    We have raw data that needs cleaning. We remove garbage values, null values, and missing values. We also remove extra columns that are not needed for our project. In the dataset, the app name column has one missing value, which we filled randomly. Categorical data is filled with mode, and numerical columns are filled with the mean. Install data initially has null values or data with “+” and “,”, so we replace these with spaces.
</p>
<pre><code>df.isnull().sum()
sns.heatmap(df.isnull(), cmap="GnBu_r")
plt.show()

df["App Name"].value_counts(dropna=False)
df["App Name"].isnull().sum()
df.sort_values("App Name", ascending=True, inplace=False)
df["App Id"].value_counts(dropna=False)
df["App Id"].isnull().sum()
df["Category"].value_counts(dropna=False)
df["Category"].isnull().sum()

df[df["Category"] == "Education"]
df[df["Category"] == "Music & Audio"]
df[df["Category"] == "Entertainment"]
df[df["Category"] == "Books & Reference"]
df[df["Category"] == "Personalization"]

