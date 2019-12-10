"""

ball_in_play.py

"""



import os
import datetime
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import Tracab as tracab
import Tracking_Visuals as vis
import Tracking_Velocities as vel
import Tracking_Fatigue as fatigue
import helpers


# config
current_dir = os.path.dirname(os.getcwd())
dir_path = os.path.join(current_dir, 'Aalborg_Jeff')
cluster_dir_path = '/n/home03/lshaw/Tracking/Tracab/SuperLiga/'
LEAGUE = 'DSL'
PLAYER_ID_to_JERSEY_NUM_LOC = '../playerid_jerseynum_map.csv'

# all games data
all_Aalborg_games = [x for x, _, _ in os.walk(dir_path) if x.count('_TracDAT')]
all_cluster_games = [x for x, _, _ in os.walk(cluster_dir_path) if x.split('/')[-1].isnumeric()]
all_game_ids = list(map(int, [path.split('/')[-1] for path in all_cluster_games]))
game_ids_w_player_mapping = pd.read_csv(PLAYER_ID_to_JERSEY_NUM_LOC)['Match ID'].unique()


# LONG RUN: for each game, get time series of energy expenditure
games = all_cluster_games
for path in games:
#     match_id, _ = path.replace(os.path.dirname(path) + '/', '').split('_') # local Aalborg
    match_id = path.split('/')[-1] # cluster
    
    # skip games which are not in player id mapping with player info
    if int(match_id) not in game_ids_w_player_mapping:
        print('skipped match {}'.format(match_id))
        continue
    
    # read 
    fpath, fname = path, str(match_id)
    frames_tb, match_tb, team1_players, team0_players = tracab.read_tracab_match_data(LEAGUE, fpath,
                                                                                      fname, verbose=True)
    
    ball_series = pd.DataFrame([[x.ball_status, x.ball_team] for x in frames_tb], 
                           columns=['ball_status', 'ball_team'])
    
    # save
    OUTFILE_NAME = 'ball_in_play_{}.csv'.format(match_id)
    ball_series.to_csv(os.path.join('../saved', OUTFILE_NAME))
        
    # free memory
    del team1_players, team0_players, match_tb, frames_tb
    