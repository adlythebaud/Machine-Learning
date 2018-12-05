#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:09:46 2018

@author: adlythebaud
"""

import pandas as pd

# 1. define where we will get our data from.
# This is from the "Million Songs Dataset". Contains data from last.fm, musixmatch, thisismyjam etc..
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

# 2. import triplets data into dataframe
song_df_1 = pd.read_table(triplets_file,header=None, nrows = 100000)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

# 3. import songs_metadata_file into dataframe
song_df_2 =  pd.read_csv(songs_metadata_file, nrows = 100000)

# 4. merge dataframes
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

# 5 Data Transformation
 
# group dataframe by song, aggregate listen counts.
song_grouped = song_df.groupby(['title']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'title'], ascending = [0,1])

# 6. Build recommender system

# count number of unique users
users = song_df['user_id'].unique()
len(users)
songs = song_df['title'].unique()
len(songs)

