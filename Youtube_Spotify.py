#!/usr/bin/env python
# coding: utf-8

# # About Dataset
# 
# **Dataset of songs of various artist in the world and for each song is present:**
# 
# **Several statistics of the music version on spotify, including the number of streams;**
# 
# **Number of views of the official music video of the song on youtube.**

# ## Content¶
# 
# It includes 26 variables for each of the songs collected from spotify. These variables are briefly described next:
# 
# **Track**: name of the song, as visible on the Spotify platform.
# 
# **Artist**: name of the artist.
# 
# **Url_spotify**: the Url of the artist.
# 
# **Album**: the album in wich the song is contained on Spotify.
# 
# **Album_type**: indicates if the song is relesead on Spotify as a single or contained in an album.
# 
# **Uri**: a spotify link used to find the song through the API.
# 
# **Danceability**: describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
# 
# **Energy**: is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
# 
# **Key**: the key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
# 
# **Loudness**: the overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
# 
# **Speechiness**: detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
# 
# **Acousticness**: a confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
# 
# **Instrumentalness**: predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
# 
# **Liveness**: detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
# 
# **Valence**: a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
# 
# **Tempo**: the overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
# 
# **Duration_ms**: the duration of the track in milliseconds.
# 
# **Stream**: number of streams of the song on Spotify.
# 
# **Url_youtube**: url of the video linked to the song on Youtube, if it have any.
# 
# **Title**: title of the videoclip on youtube.
# 
# **Channel**: name of the channel that have published the video.
# 
# **Views**: number of views.
# 
# **Likes**: number of likes.
# 
# **Comments**: number of comments.
# 
# **Description**: description of the video on Youtube.
# 
# **Licensed**: Indicates whether the video represents licensed content, which means that the content was uploaded to a channel linked to a YouTube content partner and then claimed by that partner.
# 
# **official_video**: boolean value that indicates if the video found is the official video of the song. 
#     
# #### NOTE :
# **These datas are heavily dependent on the time they were collected, which is in this case the 7th of February, 2023**            

# # Importing Libraries and Dataset

# In[1]:


import pandas as pd 
import random
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")
sns.set_palette("pastel")
sns.set_context("notebook", font_scale=1.2)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# In[2]:


df = pd.read_csv('spotify_youtube.csv')
df.sample(5)


# In[50]:


df.shape # dataset contains 19550 rows and 22 columns


# In[54]:


df.size # this is the memory of the dataset in bytes


# # Data Cleaning

# #### To streamline the data for analysis, we'll need to remove unnecessary columns like "Unnamed:0", "Url_spotify", "Uri", "Url_youtube", and any others that aren't relevant. This will help us focus on the important data for our analysis.

# In[3]:


df.columns


# In[4]:


df.drop(columns=['Unnamed: 0','Url_spotify','Uri','Url_youtube','Description','Title'],inplace=True)


# DataSet after deleting columns that are not useful.

# In[5]:


df.sample(5)


# ### displaying summary information of the dataset

# In[6]:


df.info()


# we have 21 columns out of which 7 are object/string and 14 are float/numerical

# In[7]:


null_val = df.isnull().sum()
print(null_val)


# In[8]:


# Calculate the momentum of missing values for each column
missing_values_momentum = (df.isna().sum() / len(df)) * 100


# In[9]:


missing_values_momentum.plot(kind='bar',color='red')
plt.title('Momentum of Missing Values for Each Column')
plt.xlabel('Columns')
plt.ylabel('Percentage of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[10]:


columns_with_less_missing = ['Danceability','Energy','Key','Loudness','Speechiness','Acousticness'
                             ,'Instrumentalness','Liveness','Valence','Tempo','Duration_ms']

for column in columns_with_less_missing:
    df[column].fillna(df[column].mean(),inplace=True)


# Since these columns have only 2 missing values each, imputing the missing values with the mean or median of the respective column is a reasonable approach. This method helps to preserve the overall distribution of the data and minimizes the loss of information.

# In[11]:


columns_with_many_missing = ['Channel','Views','Likes','Comments','Licensed','official_video','Stream']

for column in columns_with_many_missing:
    df.dropna(subset=[column],inplace = True)


# Deleting rows with missing values may be a suitable option if the missing values represent a small portion of the dataset and removing them does not significantly impact the analysis. However, this approach can lead to loss of information, especially if the missing values are not randomly distributed.

# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# Now all datatypes are correct and all null values have either been dropped or have been replaced by there mean

# ### Select only the numerical columns for descriptive statistics

# In[14]:


numerical_columns = df.select_dtypes(include=['float64'])
numerical_columns.describe().T


# ### Observational Findings:
# 
# In terms of **vocalization**, there's a discernible trend towards a more subdued presence (0.095 on average), hinting at a preference for instrumental arrangements over vocal-centric compositions. Yet, this trend isn't uniform, showcasing a broad spectrum of vocalization frequencies.
# 
# Regarding **acoustic elements**, the dataset leans towards a moderate embrace (0.289 on average), suggesting a substantial incorporation of acoustic nuances within the musical landscape. This inclination, however, manifests in a tapestry of diverse acoustic textures, reflecting a rich mosaic of sonic expressions.
# 
# The **instrumental essence** resonates modestly (0.055 on average), underscoring a prevailing inclination towards vocal accompaniment. Nevertheless, instances of pure instrumental compositions punctuate the dataset, denoted by pronounced instrumental characteristics.
# 
# **Live performance** quality exhibits a nuanced balance (0.191 on average), suggesting a commendable fusion of studio polish and live authenticity. This equilibrium, however, varies across the dataset, with certain tracks exuding heightened live energy.
# 
# **Emotionally**, the musical fabric gravitates towards a moderate positivism (0.529 on average), imbuing compositions with an uplifting aura. Yet, this emotional landscape isn't monolithic, showcasing a kaleidoscope of emotional hues.
# 
# Rhythmic cadences unfold at a **moderately paced tempo** (approximately 120.61 BPM on average), underscoring a rhythmic diversity that ebbs and flows across the musical terrain.
# 
# In terms of **duration**, compositions span a wide temporal spectrum, with an average duration of approximately 224,628 milliseconds. This temporal breadth underscores the diversity in compositional structures, ranging from succinct vignettes to expansive opuses.
# 
# Metrics gauging **popularity and engagement** exhibit a kaleidoscope of fluctuations, underscoring the multifaceted nature of audience interaction. While some compositions bask in the limelight of high engagement, others dwell in the shadows of lesser interaction.

# In[ ]:





# # Exploratory Data Analysis

# In[ ]:





# In[55]:


# Select numerical columns
numerical_columns = df.select_dtypes(include=['float64']).columns

# Plot histograms for each numerical feature
for column in numerical_columns:
    
    colors = {'gold', '#00FF00', '#FF5733', 'lightgreen', 'skyblue'}
    random_color = random.choice(list(colors))
    e_colors = {'white', 'brown', 'black'}
    random_e_color = random.choice(list(e_colors))
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=column, bins=30, color=random_color, edgecolor=random_e_color, kde=True)
    plt.title(f'Distribution of {column}', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()
    


# ## Observational Insights:
# 
# #### Speechiness:
# 
# The data skews towards the right, predominantly showcasing songs with minimal spoken elements. However, a notable minority features elevated speechiness, suggesting the integration of spoken word segments.
# 
# #### Acousticness:
# 
# The distribution displays a relatively uniform spread, with a concentration towards lower acousticness values. Nevertheless, a discernible cluster of songs exhibits pronounced acoustic qualities, indicating a substantial reliance on acoustic instrumentation.
# 
# #### Instrumentalness:
# 
# A prevalent skew towards lower instrumentalness values is evident, underscoring the predominance of vocal-driven compositions. Yet, a prolonged tail on the right signifies the presence of purely instrumental tracks, albeit in a minority.
# 
# #### Liveness:
# 
# The distribution tilts slightly to the right, with a majority of songs exhibiting subdued liveness attributes. Conversely, a smaller subset showcases heightened liveness, suggesting a lesser prevalence of live recordings.
# 
# #### Valence:
# 
# Symmetry characterizes the distribution, with a central tendency around moderate valence values. Nonetheless, variability persists, encapsulating a spectrum of emotional expressions ranging from positive to negative tones.
# 
# #### Tempo:
# 
# The data distribution approximates normalcy, gravitating towards a middling tempo. While outliers punctuate the distribution with extremes of tempo, the majority align with a moderate rhythmic pace.
# 
# #### Duration_ms:
# 
# A rightward skew dominates the distribution, with a peak centered on briefer durations. Despite this, a notable cohort of songs extends towards lengthier durations, indicating a diverse array of track lengths within the dataset.

# In[45]:


# Create subplots for each numerical feature
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
fig.subplots_adjust(hspace=0.5)

# List of numerical features
numerical_features = ['Speechiness', 'Acousticness', 'Instrumentalness', 
                      'Liveness', 'Valence', 'Tempo', 'Duration_ms', 
                      'Views', 'Likes', 'Comments', 'Stream']

# Plot box plots for each numerical feature
for i, feature in enumerate(numerical_features):
    row = i // 3
    col = i % 3
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']
    random_color = random.choice(list(colors))
    
    sns.boxplot(data=df[feature], ax=axs[row, col], color=random_color)
    axs[row, col].set_title(f'Distribution of {feature}', fontsize=12)
    axs[row, col].set_xlabel('')
    axs[row, col].set_ylabel('')
    axs[row, col].tick_params(axis='x', labelrotation=45)

# Remove empty subplots
for i in range(len(numerical_features), len(axs.flatten())):
    fig.delaxes(axs.flatten()[i])

# Show the plot
plt.tight_layout()
plt.show()


# ## Observational Summary:
# 
# #### Speechiness:
# Predominantly, songs exhibit low speechiness, yet some outliers feature higher spoken word content.
# 
# #### Acousticness:
# The dataset showcases a diverse range of acoustic qualities, from electronic to natural sounds.
# 
# #### Instrumentalness:
# While vocals dominate, notable exceptions highlight instrumental tracks or sections within songs.
# 
# #### Liveness:
# Variability exists, with some songs reflecting live recordings alongside studio productions.
# 
# #### Valence:
# Emotional tones span both positive and negative realms, albeit slightly favoring positivity.
# 
# #### Tempo:
# Rhythmic diversity abounds, with most songs maintaining a moderate tempo.
# 
# #### Duration_ms:
# Tracks primarily trend towards shorter durations, but outliers extend to longer compositions.
# 
# #### Engagement Metrics:
# Wide-ranging values signify diverse popularity and engagement levels, with notable outliers indicating exceptional engagement.

# In[17]:


# Scatter plot of Views vs. Likes
plt.figure(figsize=(10, 6))
plt.scatter(df['Views'], df['Likes'], color='skyblue', alpha=0.7, edgecolors='black')
plt.title('Relationship between Views and Likes')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.grid(True, linestyle='--', alpha=0.7)

# Adding a trendline
z = np.polyfit(df['Views'], df['Likes'], 1)
p = np.poly1d(z)
plt.plot(df['Views'],p(df['Views']),"r--")

plt.show()


# In[18]:


# Scatter plot of Views vs. Comments
plt.figure(figsize=(10, 6))
plt.scatter(df['Views'], df['Comments'], color='lightgreen', alpha=0.7, edgecolors='black')
plt.title('Relationship between Views and Comments')
plt.xlabel('Views')
plt.ylabel('Comments')
plt.grid(True, linestyle='--', alpha=0.7)

# Adding a trendline
z = np.polyfit(df['Views'], df['Comments'], 1)
p = np.poly1d(z)
plt.plot(df['Views'],p(df['Views']),"r--")

plt.show()


# In[19]:


# Scatter plot of Duration_ms VS stream on spotify
plt.figure(figsize=(10, 6))
plt.scatter(x=df['Duration_ms'] , y=df['Stream'], color='gold', alpha=0.7, edgecolors='black')
plt.title('Relationship between Duration_ms and Stream',fontsize=16)
plt.xlabel('Duration_ms')
plt.ylabel('Stream in millions')

# Adding a trendline
z = np.polyfit(df['Duration_ms'], df['Stream'], 1)
p = np.poly1d(z)
plt.plot(df['Duration_ms'],p(df['Duration_ms']),"r--")

plt.show()


# In[20]:


plt.subplots(figsize=(10,6))
sns.scatterplot(x=df['Danceability'],y=df['Energy'],hue=df['Loudness'],alpha=0.7)
plt.title('Relationship Between Energy , Danceability and Loudness',fontsize=16)

# Adding a trendline
z = np.polyfit(df['Danceability'], df['Energy'], 1)
p = np.poly1d(z)
plt.plot(df['Danceability'],p(df['Danceability']),"r--")

plt.show()


# In[23]:


# Convert boolean values to strings ('True' and 'False')
df['official_video'] = df['official_video'].astype(str)

# Set the style of the seaborn plot
sns.set(style="whitegrid")

# Create a violin plot comparing the distribution of 'Speechiness' between official and non-official videos
plt.figure(figsize=(10, 6))
sns.violinplot(x='official_video', y='Speechiness', data=df, palette={'True': 'lightgreen', 'False': 'lightcoral'})
plt.title('Distribution of Speechiness between Official and Non-Official Videos', fontsize=16)
plt.xlabel('Official Video', fontsize=14)
plt.ylabel('Speechiness', fontsize=14)

# Show the plot
plt.show()


# In[24]:


num_col = df.select_dtypes(include=['float64'])

plt.subplots(figsize=(10,8))
sns.heatmap(num_col.corr(), linecolor='black', linewidths=0.4)
plt.title('Correlation between Numerical Columns',fontsize=18)

plt.show()


# In[ ]:





# In[69]:


labels = ('Licensed & Official', 'Not Licensed & Not Official', 'Official but Not Licensed')
colors = ['orange', 'pink', 'lightgreen']  

plt.pie(df[['Licensed', 'official_video']].value_counts().tolist(), colors=colors, labels=labels
        , autopct='%1.1f%%',explode = [0,0.1,0])
plt.title('Share of Licensed and Official Videos',fontsize=16)

plt.tight_layout()
plt.show()


# In[71]:


album_type_counts = df['Album_type'].value_counts()

colors = ['gold', 'lightgreen', 'lightpink']
explode = [0,0.,0.05]
label=['Album','Single','Compilation']

album_type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=120, colors=colors, explode=explode , labels=label)
plt.title('Number of Songs by Album Type',fontsize=16)


plt.tight_layout()
plt.show()


# In[ ]:





# ## Creating a functions to compare different columns 

# ### (1) Function groups dataframe based on a single column and then creates a dual y-axis graph to show averages .
# ### grouped_mean()
# The function takes 6 parameters
# 
# 1) grouping column
# 2) first y-axis column
# 3) second y-axis column
# 4) sorting column
# 5) ascending(True/False)
# 6) number of records from top
# 

# In[26]:


def grouped_mean(grouping_column='Artist',first_y='Likes',second_y='Comments',sortby_col='Likes',ascend=False,top=10,):
    
    grouped = df.groupby(by=[grouping_column])
    num_col = df.select_dtypes(include=['float64'])
    df1 = num_col.groupby(df[grouping_column]).mean()
    
    colors = {'gold', '#00FF00', '#FF5733', 'skyblue', 'pink'}
    random_color = random.choice(list(colors))
    rdf = pd.DataFrame(df1.sort_values(by = sortby_col , ascending=ascend ).head(top)[[first_y,second_y]])
    
    if ascend == False:
        plt.subplots(figsize=(10,6))
        plt.title(f"Top {top} {grouping_column} by Average {sortby_col}")
        sns.barplot(x=rdf.index, y=rdf[first_y], color=random_color, label=first_y)
        sns.barplot(x=rdf.index, y=rdf[second_y], color='Brown', label=second_y, width=0.6)
        plt.xticks(rotation=90)
        plt.xlabel(grouping_column)
        plt.ylabel(f"{first_y} and {second_y}")
        plt.legend()
        plt.show()
        
    elif ascend == True:
        plt.subplots(figsize=(10,6))
        plt.title(f"Bottom {top} {grouping_column} by Average {sortby_col}")
        sns.barplot(x=rdf.index, y=rdf[sortby_col], color=random_color)
        plt.xticks(rotation=90)
        plt.xlabel(grouping_column)
        plt.ylabel(f"{sortby_col}")
        plt.show()
    


# In[27]:


grouped_mean()


# In[28]:


grouped_mean(grouping_column='Album',first_y='Views',second_y='Stream',sortby_col='Views',ascend=False,top=10)


# In[29]:


grouped_mean(grouping_column='Album',ascend=True,top=20)


# In[30]:


grouped_mean(grouping_column='Album_type',top=3)


# ### (2) Function groups dataframe based on a single column and then creates a dual y-axis graph to show averages
# ### grouped_sum()

# It does everything in the same way as the earlier function . It just sums rather than taking averages .

# In[31]:


def grouped_sum(grouping_column='Artist',first_y='Likes',second_y='Comments',sortby_col='Likes',ascend=False,top=10,):
    
    grouped = df.groupby(by=[grouping_column])
    num_col = df.select_dtypes(include=['float64'])
    df1 = num_col.groupby(df[grouping_column]).sum()
    
    colors = {'gold', '#00FF00', '#FF5733', 'skyblue', 'pink'}
    random_color = random.choice(list(colors))
    rdf = pd.DataFrame(df1.sort_values(by = sortby_col , ascending=ascend ).head(top)[[first_y,second_y]])
    
    if ascend == False:
        plt.subplots(figsize=(10,6))
        plt.title(f"Top {top} {grouping_column} by Total {sortby_col}")
        sns.barplot(x=rdf.index, y=rdf[first_y], color=random_color, label=first_y,width=0.6)
        sns.lineplot(x=rdf.index, y=rdf[second_y], color='black', label=second_y, marker='o')
        plt.xticks(rotation=90)
        plt.xlabel(grouping_column)
        plt.ylabel(f"{first_y} and {second_y}")
        plt.legend()
        plt.show()
        
    elif ascend == True:
        plt.subplots(figsize=(10,6))
        plt.title(f"Bottom {top} {grouping_column} by Total {sortby_col}")
        sns.barplot(x=rdf.index, y=rdf[sortby_col], color=random_color)
        plt.xticks(rotation=90)
        plt.xlabel(grouping_column)
        plt.ylabel(f"{sortby_col}")
        plt.show()
    


# In[32]:


grouped_sum()


# In[33]:


grouped_sum(grouping_column='Album',first_y='Views',second_y='Stream',sortby_col='Views')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




