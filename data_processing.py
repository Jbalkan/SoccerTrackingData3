"""

copy of data processing bits of notebooks

by Jeff Balkanski

"""

import os
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Tracab as tracab
import Tracking_Visuals as vis
import Tracking_Velocities as vel
import Tracking_Fatigue as fatigue
import helpers
import importlib



def update_pkl():
    """ add ball in play data to pkl files"""
    
    # get paths
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data_processed')
    all_data_processed = [os.path.join(data_path, f) for f in os.listdir(data_path)]

    # match pkl and csv files
    csvs = [f for f in all_data_processed if f.find('.csv') != -1]
    game_dicts = {f.split('/')[-1].split('.')[0].split('_')[-1]: f for f in all_data_processed 
                  if f.find('.pkl') != -1}
    matching_csvs = {path.split('_')[-1].split('.')[0]: path for path in csvs 
                     if path.split('_')[-1].split('.')[0] in game_dicts.keys()}

    # get ball in play time series
    for game_id, path in matching_csvs.items():
        df = pd.read_csv(path, index_col=0)

        # read pkl
        with open(os.path.join(game_dicts[game_id]), 'rb') as infile:
            game_data = pickle.load(infile)
        game_data['ball_in_play'] = df.values

        # re write
        with open(os.path.join(game_dicts[game_id]), 'wb') as infile:
            pickle.dump(game_data, infile)
