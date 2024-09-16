# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Reading the dataset
data = pd.read_csv("Instagram data.csv", encoding='latin1')

# Previewing the dataset
print(data.head())

# Checking for null values and dropping rows with missing values
data.isnull().sum()
data = data.dropna()

# Information about the dataset
data.info()

# Visualizing the distribution of 'From Home' impressions
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'], kde=True)
plt.show()

# Visualizing the distribution of 'From Hashtags' impressions
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'], kde=True)
plt.show()

# Visualizing the distribution of 'From Explore' impressions
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.histplot(data['From Explore'], kde=True)
plt.show()

# Summing the different sources of impressions
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

# Pie chart for impressions from various sources
labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

# Generating a word cloud for Captions
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.style.use('classic')
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Generating a word cloud for Hashtags
text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Scatter plot to visualize the relationship between Likes and Impressions
figure = px.scatter(data_frame=data, x="Impressions", y="Likes", size="Likes", 
                    trendline="ols", title="Relationship Between Likes and Impressions")
figure.show()

# Scatter plot to visualize the relationship between Comments and Impressions
figure = px.scatter(data_frame=data, x="Impressions", y="Comments", size="Comments", 
                    trendline="ols", title="Relationship Between Comments and Total Impressions")
figure.show()

# Scatter plot to visualize the relationship between Shares and Impressions
figure = px.scatter(data_frame=data, x="Impressions", y="Shares", size="Shares", 
                    trendline="ols", title="Relationship Between Shares and Total Impressions")
figure.show()

# Scatter plot to visualize the relationship between Saves and Impressions
figure = px.scatter(data_frame=data, x="Impressions", y="Saves", size="Saves", 
                    trendline="ols", title="Relationship Between Post Saves and Total Impressions")
figure.show()

# Conversion rate (Follows/Profile Visits)
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print("Conversion Rate: ", conversion_rate)

# Scatter plot to visualize the relationship between Profile Visits and Follows
figure = px.scatter(data_frame=data, x="Profile Visits", y="Follows", size="Follows", 
                    trendline="ols", title="Relationship Between Profile Visits and Followers Gained")
figure.show()

# Preparing data for the model
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])

# Splitting the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Building the model
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)

# Printing the model score on test data
print("Model Score: ", model.score(xtest, ytest))

# Predicting impressions using sample features
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
prediction = model.predict(features)
print("Predicted Impressions: ", prediction)
