"""

Metrics.py

@author: Jeff Balkanski

"""

import numpy as np
import pandas as pd


#############################################################
#                                                           #
#                       Abstractions                        #
#                                                           #
#############################################################


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


def get_all_values(player, metric):
    return np.array([eval('t.' + metric) for t in player.frame_targets])


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


#############################################################
#                                                           #
#                       Metrics                             #
#                                                           #
#############################################################


def distance(start, end):
    xs, ys = start
    xe, ye = end
    return np.sqrt((xe-xs) ** 2 + (ye-ys) ** 2)


def get_total_distance(player, i_start=0, i_end=None):
    """ get total distance ran by a player

    Arguments:
        player {tracab player} -- player to analyze

    Keyword Arguments:
        i_start {int} -- starting frame id (default: {0})
        i_end {[type]} -- ending frame id (default: {None})

    Returns:
        float -- total distance
    """
    total_distance = 0
    i_end = i_end if i_end else len(player.frame_timestamps)
    for i in range(i_start, i_end - 1):
        start = (player.frame_targets[i].pos_x, player.frame_targets[i].pos_y)
        end = (player.frame_targets[i + 1].pos_x, player.frame_targets[i + 1].pos_y)
        total_distance += distance(start, end)

    return round(total_distance/100, 2)


def get_total_time(player):
    total_time = 0
    ts_lst = player.frame_timestamps
    start = ts_lst[0]
    for i in range(len(ts_lst) - 1):
        # find end of a period
        if ts_lst[i + 1] < ts_lst[i]:
            total_time += ts_lst[i] - start
            start = ts_lst[i + 1]

    total_time += (ts_lst[-1] - start)

    return total_time


def top_speed(player, i_start, i_end, source='filter', unit='ms'):
    if source == 'filter':
        top_speed = max([player.frame_targets[i].speed_filter for i in range(i_start, i_end)])
    elif source == 'data':
        top_speed = max([player.frame_targets[i].speed for i in range(i_start, i_end)])
    else:
        raise Exception
    
    if unit == 'ms':
        return round(top_speed, 2)
    elif unit == 'kmh':
        return round(top_speed * 3.6, 2)


def mean_speed(player, i_start, i_end, source='filter', unit='ms'):
    if source == 'filter':
        top_speed = max([player.frame_targets[i].speed_filter for i in range(i_start, i_end)])
    elif 'source' == 'data':
        top_speed = np.mean([player.frame_targets[i].speed for i in range(i_start, i_end)])
    else:
        raise Exception

    if unit == 'ms':
        return round(top_speed, 2)
    elif unit == 'kmh':
        return round(top_speed * 3.6, 2)
