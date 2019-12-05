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


# helpers
def get_time_series(players_full_game):
    metric_values = []
    for row in players_full_game.to_dict(orient='records'):
        player = row['obj']
        val_array = fatigue.get_energy_expenditure(player)
        metric_values.append([row['match_id'], row['player_id'], val_array])
    EX_all = np.array(metric_values)
    
    return EX_all

# all games data
data_dict = {}
all_Aalborg_games = [x for x, _, _ in os.walk(dir_path) if x.count('_TracDAT')]
all_cluster_games = [x for x, _, _ in os.walk(cluster_dir_path) if x.split('/')[-1].isnumeric()]
all_cluster_games.sort(reverse=True)
all_game_ids = list(map(int, [path.split('/')[-1] for path in all_cluster_games]))

# player data
game_ids_w_player_mapping = pd.read_csv(PLAYER_ID_to_JERSEY_NUM_LOC)['Match ID'].unique()
print('Sucessfully read data paths and player data')

# LONG RUN: for each game, get time series of energy expenditure
games = all_Aalborg_games
for path in games:
    game_data = {}
    
    match_id, _ = path.replace(os.path.dirname(path) + '/', '').split('_') # local Aalborg
#     match_id = path.split('/')[-1] # cluster
    print('Starting energy expenditure calculcation for game {} at location {}'.format(match_id, path))

    # skip games which are not in player id mapping with player info
    if int(match_id) not in game_ids_w_player_mapping:
        print('skipped match {}'.format(match_id))
        continue
    
    # store path
    game_data['data_path'] = path
    data_dict[match_id] = {'data_path': path}
    
    # read 
    fpath, fname = path, str(match_id)
    frames_tb, match_tb, team1_players, team0_players = tracab.read_tracab_match_data(LEAGUE, fpath,
                                                                                      fname, verbose=True)
    
    # players info
    players_info = helpers.map_playerids_positions(team1_players, team0_players, 
                                                    match_id, loc_mapping=PLAYER_ID_to_JERSEY_NUM_LOC)
    
    # get stats
    time_series_mat = get_time_series(players_info)
    
    # store 
    game_data['player_info'] = players_info.drop('obj', axis=1)
    game_data['energy_x'] = time_series_mat
    data_dict[match_id]['player_info'] = players_info.drop('obj', axis=1)
    data_dict[match_id]['energy_x'] = time_series_mat
    
    # save
    OUTFILE_NAME = 'EX_time_series_game_{}.pkl'.format(match_id)
    with open(os.path.join('../saved', OUTFILE_NAME), 'wb') as outfile:
        pickle.dump(data_dict, outfile)
        
    # free memory
    del team1_players, team0_players, match_tb, frames_tb
    
print('successfully computed time series for all games')
