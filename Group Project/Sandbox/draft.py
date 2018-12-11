#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:56:52 2018

@author: adlythebaud
"""

import sys
import spotipy
import spotipy.util as util

scope = 'user-top-read'

def getUserTopTracks(sp):
    topTracks = sp.current_user_top_tracks(time_range='long_term')
    print(topTracks['items'])
    pass

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage :{0} username".format(sys.argv[0]))
    sys.exit()

token = util.prompt_for_user_token(username,
    scope=scope,
    client_id='770b0b660a334803a9ec35ee7079eb71',
    client_secret='0d9ee017af104880b73b656364f1c11f',
    redirect_uri="http://localhost/")

if token:
    print("Token Received! ")

    sp = spotipy.Spotify(auth=token)

    getUserTopTracks(sp)

else:
    print("Token was not received")



#import spotipy
#import sys
#import spotipy.util
#
#
#
#
#username = "121570006"
#scope = "user-top-read"
#client_id = "http://localhost:8888/callback"
##client_secret = 
#
#util.prompt_for_user_token(username, scope, client_id = "Spotty://returnAfterLogin")
