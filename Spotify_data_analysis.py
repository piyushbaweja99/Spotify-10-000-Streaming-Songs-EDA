#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Anaysis of Top 10000 streamed spotify songs
# 
# The following Dataset is having data for the top 10000 songs streamed on spotify  inclusive of there positions, total streams and peak streams.

# ## Downloading the Dataset

# In[1]:


dataset_url = 'https://www.kaggle.com/datasets/rakkesharv/spotify-top-10000-streamed-songs' 


# In[2]:


get_ipython().system(' pip install opendatasets --upgrade --quiet')


# In[3]:


import opendatasets as od
od.download(dataset_url)


# In[4]:


data_dir = './spotify-top-10000-streamed-songs/'


# In[5]:


import os
os.listdir(data_dir)


# In[6]:


import warnings
warnings.filterwarnings('ignore')


# ## Data Preparation and Cleaning
# 
# In this step we will clean data as per our need and what we need to analyse.
# 
# To start analysing we will import all the necessary packages and start by reading csv using pandas
# 
# 

# In[7]:


# importing all necessary packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[8]:


spotify_df = pd.read_csv('spotify-top-10000-streamed-songs/Spotify_final_dataset.csv')
spotify_df


# In[9]:


spotify_df.info() # Checking Dtypes of columns


# In[10]:


spotify_df.rename(columns = {'Postion': 'position'}, inplace =True)
spotify_df.rename(columns = {'Artist Name': 'artist_name'}, inplace =True)
spotify_df.rename(columns = {'Song Name': 'song_name'}, inplace =True)
spotify_df.rename(columns = {'Days': 'days'}, inplace =True)
spotify_df.rename(columns = {'Top 10 (xTimes)': 'top_10_times'}, inplace =True)
spotify_df.rename(columns = {'Peak Position': 'peak_position'}, inplace =True)
spotify_df.rename(columns = {'Peak Position (xTimes)': 'peak_position_times'}, inplace =True)
spotify_df.rename(columns = {'Peak Streams': 'peak_streams'}, inplace =True)
spotify_df.rename(columns = {'Total Streams': 'total_streams'}, inplace =True)


# In[11]:


spotify_df.duplicated().sum()  # Finding duplicates in Data set


# In[12]:


spotify_df.isnull().sum()


# In[13]:


# Converted peak_position to numeric and cleared all (x..)

spotify_df.iloc[:,6] = pd.to_numeric(spotify_df.iloc[:,6].str.extract(r'(\d+)', expand=False))


# In[14]:


spotify_df.info()


# In[15]:


null_rows= spotify_df.loc[spotify_df.song_name.isnull()]
null_rows


# In[16]:


# As no song name is there need to check for duplicacy

for i in null_rows.artist_name:
    count = (spotify_df.artist_name == i).sum()
    if count > 1:
        print(f"{i} has occured {count} times")
    else:
        print(f"{i} occurred only once")


# In[17]:


spotify_df.song_name = spotify_df.song_name.fillna(spotify_df.artist_name)  # song_name = artist_name


# In[18]:


spotify_df['peak_position_times'].unique()  # Checking for all numerics


# In[19]:


spotify_df.isnull().sum()  # checking for null values


# In[20]:


song_by_artist = spotify_df.song_name + " by " + spotify_df.artist_name
song_by_artist
spotify_df['song_by_artist'] = song_by_artist
spotify_df


# ## Exploratory Analysis and Visualization
# 
# Now Data cleaning has been completed now we have to do visualization.
# will plot different graphs for different scenarios
# 
# 

# In[21]:


get_ipython().system('pip install plotly --upgrade --quiet')


# In[22]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (20, 20)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


#  ### Popular Artist among People

# In[23]:


artist_count = spotify_df.groupby('artist_name').size().sort_values(ascending=False)
top_artist = artist_count.head(25)
top_artist


# In[24]:


plt.figure(figsize=(12,12))
artist_count = spotify_df.groupby('artist_name').size().sort_values(ascending=False)
top_artist = artist_count.head(25)
sns.barplot(x=top_artist.values, y= top_artist.index);
plt.title('Top 25 Artists with most Hit Songs', size = 15)
plt.ylabel('Artists')
plt.xlabel('Number of Songs', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15);


# ####  Conclusion : By this plot we came to conclusion that Drake has the highest numbers of songs in this Dataset , which makes him most popular Artist

# ### Popular Songs with maximum Peak Position Times

# In[25]:


popular_songs = spotify_df.nlargest(25, 'peak_position_times')  # returns 10 songs with largest peak_postion times
popular_songs


# In[26]:


popular_songs = spotify_df.sort_values(by = 'peak_position_times', ascending = False).head(25)
plt.figure(figsize= (12,12))
sns.barplot(x='peak_position_times', y = 'song_name', data = popular_songs)
plt.title('Top 25 Songs with Peak Position', size = 15)
plt.ylabel('Songs')
plt.xlabel('Peak Position (xTimes)', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show;


# ####  Conclusion : By this plot we have an clear idea of top 25 popular Songs in this dataset, which is led by song rockstar by Post Malone.

# ### No. of Songs with peak Positions 1 to 10

# In[27]:


peak_position_count = spotify_df.groupby('peak_position').size().head(10)
peak_position_count


# In[28]:


plt.figure(figsize= (12,12))
sns.barplot(x = peak_position_count.index, y = peak_position_count.values)
plt.title('No. of songs by Peak Position', size = 15)
plt.ylabel('No. of Songs')
plt.xlabel('Peak Postion Attained', size = 15)


# #### Conclusion :  By looking at this plot we can clearly say that more than 175 songs have attained peak position as 1 in the top charts

# ### Songs which have been in top 10 for most number of times

# In[29]:


top_10_songs = spotify_df.nlargest(10, 'top_10_times')
top_10_songs


# In[30]:


plt.figure(figsize= (12,12))
sns.barplot(x = 'top_10_times', y = 'song_name' , data = top_10_songs)
plt.title('Top 10 Songs that occured in top 10 charts highest no. of times', size = 15)
plt.ylabel('Song Name', size = 15)
plt.xlabel('Top 10 (xTimes)', size = 15);
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# #### Conclusion :  By this plot we can see top 10 songs which are in constant top 10 Charts

#  ### Top streamed Songs

# In[31]:


top_10_streams = spotify_df.nlargest(10, 'total_streams')
top_10_streams


# In[32]:


plt.figure(figsize= (12,12))
sns.barplot(x = 'total_streams', y = 'song_name' , data = top_10_streams)
plt.title('Top 10 Streamed Songs', size = 15)
plt.ylabel('Song Name', size = 15)
plt.xlabel('Total Streams', size = 15);
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# #### we can conclude that songs that are mostly in top 10 are not likely to be most streamed , regardless of some exceptions like Sunflower by Post Malone

# ### Songs with Maximum peak Streams

# In[33]:


top_10_peak_streams = spotify_df.nlargest(10, 'peak_streams')
top_10_peak_streams


# In[34]:


plt.figure(figsize= (18,12))
sns.barplot(x = 'peak_streams', y = ('song_by_artist') , data = top_10_peak_streams)
plt.title('Top 10 Songs with Maximum peak Streams', size = 15)
plt.ylabel('Song Name', size = 15)
plt.xlabel('Peak Streams', size = 15);
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# #### We can clearly see which songs having maximum peak streams , and Taylor Swift wins with a bang having 6 songs in top 10 songs having peak streams

# # Answer following Questions

# #### Q1: Which Songs are mostly liked by people ie. having high peak streams and total streams ?
# 

# In[35]:


top_streams = spotify_df.nlargest(100, 'total_streams')


# In[36]:


top_peak_streams = spotify_df.nlargest(100, 'peak_streams')


# In[37]:


top_grossing_songs = top_streams.merge(top_peak_streams , on = 'song_name')
top_grossing_songs


# In[38]:


fig = px.scatter(top_grossing_songs, x = 'total_streams_x', y = 'peak_streams_y', text = 'song_name', width = 1000, height = 900)
fig.update_traces(textposition="bottom right")
fig.update_traces(marker_size=8)
fig.show()


# #### Conclusion : By looking at the plot we can see songs which are having most streams are having less peak streams

# #### Q2 Which Artist is mostly liked by people ie. having high peak streams and total streams ?

# In[39]:


top_artist_streams= spotify_df.nlargest(500, 'total_streams')
top_artist_peak_streams = spotify_df.nlargest(500, 'peak_streams')


# In[40]:


top_grossing_artist = top_artist_streams.merge(top_artist_peak_streams , on = 'song_name')
top_grossing_artist


# In[41]:


top_grossing_artist_count = top_grossing_artist.groupby('artist_name_x').size().sort_values(ascending=False)
top_grossing_artist_count


# In[42]:


top_grossing_artist_count_10 =top_grossing_artist_count.head(10)
top_grossing_artist_count_10


# In[43]:


palette_color = sns.color_palette('pastel')

fig = plt.pie(top_grossing_artist_count_10, colors=palette_color, labels=top_grossing_artist_count_10.index 
    ,autopct='%.0f%%', labeldistance=0.9)
plt.title('Ditribution Among top 10 Artists',
         fontsize=20,
          color="Black")
plt.show();


# #### Conclusion : By checking this plot we can se Drake is having 23 percent of hit songs in top 10 artists which is having top peaks and total streams

# #### Q3: Which are top songs with largest Average Streams per day

# In[44]:


spotify_df['avg_streams_per_day'] = spotify_df.total_streams / spotify_df.days

spotify_df


# In[45]:


top_songs_with_high_avg_streams = spotify_df.nlargest(10,'avg_streams_per_day')
top_songs_with_high_avg_streams


# In[46]:


plt.figure(figsize= (12,12))
sns.barplot(x = 'avg_streams_per_day', y = ('song_by_artist') , data = top_songs_with_high_avg_streams )
plt.title('Top 10 Songs with Maximum Average Streams per day', size = 15)
plt.ylabel('Song Name', size = 15)
plt.xlabel('Average Streams per day', size = 15);
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# #### Conclusion :  By seeing this plot we can see that the song which has highest average streams is Kill Bill

# #### Q4: What is thre relationship between Total Streams and Average Streams per day?

# In[47]:


sns.pairplot(spotify_df, x_vars=['total_streams'], y_vars=['avg_streams_per_day'], height=10)
plt.xlabel('Total Stream', size=20)
plt.ylabel('Average streams per day', size=20)
plt.show()


# #### Conclusion : We can clearly see there is a area in the map where almost 80 percent of results says new songs which are having low total streams having majority to average streams per day , hence some exceptions are there in which Avg Stream peaks for some songs and there are some songs with very high total stream but competing fairly in Average Streams segment.

# #### Q5:  Which is the latest most trending Song ?

# In[48]:


latest_songs = spotify_df.loc[spotify_df.days <4]
latest_songs


# In[49]:


top_new_streams = latest_songs.nlargest(50, 'total_streams')
top_new_peak_streams = latest_songs.nlargest(50,'peak_streams')


# In[50]:


top_trending_songs = top_new_streams.merge(top_new_peak_streams, on = 'song_by_artist').head(10)
top_trending_songs


# In[51]:


plt.figure(figsize= (12,12))
sns.barplot(x = 'total_streams_x', y = ('song_by_artist') , data = top_trending_songs )
plt.title('Top 10 latest Trending Songs', size = 15)
plt.ylabel('Song Name', size = 15)
plt.xlabel('Total Streams', size = 15);
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# #### Conclusion :  In the above plot we can clearly see that these are top 10 latest trending songs which are released 3 days back  or less and having top streams and top peak streams among users

# ## Inferences and Conclusion
# 
# We have done a lot of Data Analysis on different parameters , lets Conclude all this through a Visual representation

# #### Overall Distribution of Dataset

# In[52]:


spotify_df_distribution = spotify_df.hist(color='#1477d2', zorder=2, rwidth=0.9, grid=False);

plt.suptitle('Dataset distributions overview',fontsize=20,color="blue" );

plt.subplots_adjust(hspace=1);

spotify_df_distribution = spotify_df_distribution[0]
for x in spotify_df_distribution:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False);

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on");

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1);


# ## Conclusion we made through our EDA
# 
# #### 1 Top Songs, artist in every field 
# #### 2. Relationships between peak streams, total streams
# #### 3.  Average Streams per day for top songs
# #### 4. Distribution of Industry among top 10 artists  
# #### 5. Latest Songs which are trending
# #### 6. Found Artist which are top in there areas but not having top grossing songs and some artist which are in every list like Drake, Post Malone and Taylor Swift

# ## References and Future Work
# ### References : 
# #### 1. https://www.kaggle.com/datasets/rakkesharv/spotify-top-10000-streamed-songs
# #### 2. https://seaborn.pydata.org/index.html
# #### 3. https://pandas.pydata.org/docs/index.html
# 
# ### Future Work:
# 
# #### In this dataset we can conclude which artist is top grossing by merging data of multiple Columns, Can find all relation between all parameters like streams and avg_streams, can find average streams per artist 
