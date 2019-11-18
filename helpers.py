"""

Metrics.py

@author: Jeff Balkanski

"""

import os
import pickle
import numpy as np
import pandas as pd


#############################################################
#                                                           #
#                       Abstractions                        #
#                                                           #
#############################################################


def add_metric_to_player(team1_players, team0_players, func, metric_name, skip_start=1, skip_end=None):
    all_players = [(x[0], x[1], 1) for x in team1_players.items()] + [(x[0], x[1], 0) for x in team0_players.items()]
    for (num, player, team) in all_players:
        metric = func(player)
        skip_end = skip_end if not skip_end else - skip_end

        # store info
        for i, frame in enumerate(player.frame_targets[skip_start:skip_end]):
            setattr(frame, metric_name, metric[i])

        print('Done with player {} of team {}'.format(num, team))


def get_all_values(player, metric, start=0, skip_last=None):
    skip_last = len(player.frame_targets) if not skip_last else - skip_last
    series = np.array([eval('t.' + metric) for t in player.frame_targets[start:skip_last]])
    return series


def pickle_tracab_objects(frames, match, team1_players, team0_players, loc='./saved'):
    """ Save tracab objects locally

    Keyword Arguments:
        loc {str} -- [description] (default: {'./saved'})
    """
    for obj, filename in zip([frames, match, team1_players, team0_players],
                             ['frames', 'match', 'team1', 'team0']):
        with open(os.path.join('./saved', filename), 'wb') as outfile:
            pickle.dump(obj, outfile)


def read_pickled_tracab_objects(loc='./saved'):
    """ Read saved pickled trsacab objects

    Keyword Arguments:
        loc {str} -- path of pickled files (default: {'./saved'})

    Returns:
        lst -- list of tracab objects
    """
    with open(os.path.join('./saved', 'team1'), 'rb') as infile:
        team1_players = pickle.load(infile)
        print('Done reading pickle objet team1')

    with open(os.path.join('./saved', 'team0'), 'rb') as infile:
        team0_players = pickle.load(infile)
        print('Done reading pickle objet team0')

    with open(os.path.join('./saved', 'match'), 'rb') as infile:
        match_tb = pickle.load(infile)
        print('Done reading pickle objet match')

    with open(os.path.join('./saved', 'frames_tb'), 'rb') as infile:
        frames_tb = pickle.load(infile)
        print('Done reading pickle objet frames')

    return frames_tb, match_tb, team1_players, team0_players


def get_mean_every_k(arr, k=10):
    """ Get array of all the means for each interval of size k
        Returns an array of length k times smaller

    Keyword Arguments:
        k {int} -- [description] (default: {10})

    """
    if len(arr) % k != 0:
        arr = arr[:-(len(arr) % k)]

    return np.mean(arr.reshape(-1, k), axis=1)


def slice_np_diff(lst, n=1):
    """ function like np.diff but gives the difference between elements that
        are spaced by <n> indices

    Keyword Arguments:
        n {int} -- space, default is 1 and functions like np.diff (default: {1})

    """
    return (lst[n:] - lst[:-n])




def apply_to_all(team1_players, team0_players, func, verbose=True, metric_name='metric', **kwargs):
    """ apply to all players the func metric

    Arguments:
        team1_players {lst} -- players and jersey nums of team 1
        team0_players {[lst]} -- players and jersey nums of team 0
        func {function} -- metric function

    Keyword Arguments:
        metric_name {str} -- [description] (default: {'metric'})

    Returns:
        pd.DataFrame -- all data in dataframe (of string, must change)
    """
    all_players = [(1, x) for x in team0_players.items()] + [(0, x) for x in team1_players.items()]
    data = []
    for (team_id, (num, player)) in all_players:
        # apply metric by split
        col_names, vals = apply_by_split(player, func, metric_name, **kwargs)
        data.append(vals)

    all_data = pd.DataFrame(np.array([col_names.tolist()] + np.array(data).tolist()).T).set_index(0).T.set_index(['team_id', 'jersey_num'])

    return all_data


def apply_by_split(player, func, metric_name='metric',  **kwargs):
    """ apply metric of func by splits and add a summary by summing over all splits

    Arguments:
        player {tracab player} -- player to analyze
        func {(tracab player, int, int) -> ?} -- metric function

    Keyword Arguments:
        metric_name {str} -- [description] (default: {'metric'})
        split_size {int} -- [description] (default: {5})

    Returns:
        np.array -- all split data
    """
    # input valid
    if len(player.frame_targets) == len(player.frame_timestamps) == len(player.frameids):
        n = len(player.frame_targets)
    else:
        raise Exception('Error, number of frame')
    
    # make splits
    split_size = 5
    split_names = [i * split_size for i in range(1, int(45/split_size) + 1)] + ['additional_1'] \
                    + [i * split_size + 45 for i in range(1, int(45/split_size) + 1)] + ['additional_2']
    data = [[s] for s in split_names]
    ts_lst = player.frame_timestamps

    # for every split
    i_start, ts_start, i_split = 0, 0, 0
    for i in range(n - 1):
        # end of a split
        if (ts_lst[i] - ts_start) >= split_size:
            data[i_split].append(func(player, i_start, i_end=i, **kwargs))
            i_start, ts_start = i, ts_lst[i]
            i_split += 1

        # end of a period
        if ts_lst[i + 1] < ts_lst[i]:
            # additional time
            data[i_split].append(func(player, i_start, i_end=i, **kwargs))
            i_start, ts_start = i, 0
            i_split += 1

    # second additional time
    data[i_split].append(func(player, i_start, i_end=i, **kwargs))
    i_start, ts_start = i, ts_lst[i]

    # sum up for summary
    data.append(['total', np.round(np.sum([x[1] for x in data]), 2)])

    return np.array([['team_id', player.teamID], ['jersey_num', player.jersey_num]] + data).T


