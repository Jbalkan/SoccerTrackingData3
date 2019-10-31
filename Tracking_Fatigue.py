"""

Tracking_Fatigue.py

written by Jeff Balkanski

"""

import numpy as np

import helpers


###################################
#                                 #
#           Metrics               #
#                                 #
###################################


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





def estimate_player_VeBDA(team1_players, team0_players):
    """ estimate player energy expenditure as Vector Dynamic Body Acceleration,
        using the sum of  acceleration vectors, stored as a_filter in tracab
        frames. Modifies inplace.

    Returns:
        None --
    """
    # get for all players
    all_players = list(team0_players.items()) + list(team1_players.items())
    for (num, player) in all_players:
        VeBDA = np.cumsum(helpers.get_all_values(player, 'a_filter', start=1))

        # store info
        for i, frame in enumerate(player.frame_targets[1:]):
            frame.VeBDA = VeBDA[i]


def estimate_metabolic_power(team1_players, team0_players):
    """ Estimate all player's metabolic power following the method of Cristian Osgnach and 
        di Prampero. Energy cost and metabolic power in elite soccer, 2010

    Arguments:
        team1_players {dict} -- [description]
        team0_players {dict} -- [description]
    """
    # get for all players
    all_players = list(team0_players.items()) + list(team1_players.items())
    for (num, player) in all_players:

        # get accelerations
        ax = helpers.get_all_values(player, 'ax', start=1)
        ay = helpers.get_all_values(player, 'ay', start=1)
        vx = helpers.get_all_values(player, 'vx', start=1)
        vy = helpers.get_all_values(player, 'vy', start=1)

        # in m/s^2
        g = 9.81

        # find angle, of equivalent slope
        ix = np.tan(np.pi/2. - np.arctan(g/ax))
        iy = np.tan(np.pi/2. - np.arctan(g/ay)) 

        # find energy cost
        EC_x = 155.4*ix**5 - 30.4*ix**4 - 43.3*ix**3 + 46.3*ix**2 + 19.5*ix + 3.6
        EC_y = 155.4*iy**5 - 30.4*iy**4 - 43.3*iy**3 + 46.3*iy**2 + 19.5*iy + 3.6

        # equivalent mass ratio
        mx = np.sqrt((np.power(ax, 2)/g**2) + 1)
        my = np.sqrt((np.power(ay, 2)/g**2) + 1)

        # matabolic power
        Px = np.multiply(EC_x, mx, vx)
        Py = np.multiply(EC_y, my, vy)

        metabolic_power = Px + Py

        # store
        for i, frame in enumerate(player.frame_targets[1:]):
            frame.metabolic = metabolic_power[i]
