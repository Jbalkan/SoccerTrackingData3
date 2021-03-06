# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:44:50 2019

Module for measuring ball and player velocity (and acceleration, eventually) from tracking data 

@author: laurieshaw
"""

import numpy as np
import scipy
import scipy.signal as signal
# from scipy.signal import butter, lfilter, freqz


def smooth_ball_position(frames, match,_filter='Savitzky-Golay', window=5, polyorder=3):
    # For smoothing ball positions. Not used.
    for half in [1,2]:
        istart = match.period_attributes[half]['iStart']
        iend = match.period_attributes[half]['iEnd']
        nframes = iend-istart+1
        r = np.nan*np.zeros((nframes,3),dtype=float)
        mask = []
        count = 0
        print( istart, iend)
        for frame in frames[istart:iend+1]:
            frame.ball_pos_x_smooth = np.nan
            frame.ball_pos_y_smooth = np.nan
            frame.ball_pos_z_smooth = np.nan
            if frame.ball:
                mask.append(count)
                r[count,:] = np.array( [frame.ball_pos_x,frame.ball_pos_y,frame.ball_pos_z] ) 
                count += 1
        r_raw = r.copy()
        r[:,0] = signal.savgol_filter(r[:,0],window_length=window,polyorder=polyorder)
        r[:,1] = signal.savgol_filter(r[:,1],window_length=window,polyorder=polyorder)
        r[:,2] = signal.savgol_filter(r[:,2],window_length=window,polyorder=polyorder)
        for i in mask:
            if frames[istart+i].ball:
                frames[istart+i].ball_pos_x_smooth = r[i,0]
                frames[istart+i].ball_pos_y_smooth = r[i,1]
                frames[istart+i].ball_pos_z_smooth = r[i,2]

    return frames, r, r_raw


def estimate_ball_velocities(frames,match,_filter='Savitzky-Golay',window=5,polyorder=3,maxspeed=40):
    # Apply filter to ball displacement between frames to get a smooth estimate of velocity
    x = []
    xraw = []
    for half in [1,2]: # do each half seperately
        istart = match.period_attributes[half]['iStart'] # first frame in half
        iend = match.period_attributes[half]['iEnd'] # last frame in half
        nframes = iend-istart+1
        dr = np.nan*np.zeros((nframes,3),dtype=float)
        dt = np.nan*np.zeros(nframes,dtype=float)
        mask = []
        count = 0
        print (istart, iend)
        for frame in frames[istart:iend+1]:
            frame.ball_speed_filter3 = np.nan # in 3d
            frame.ball_speed_filter2 = np.nan # in the x-y plane only
            # The ball isn't necessarily in every frame, so figure out which frames it is in
            if frame.ball:
                mask.append(count)
                dr[count,:] = np.array( [frame.ball_pos_x,frame.ball_pos_y,frame.ball_pos_z] )/100. # to m
                dt[count] = frame.timestamp*60 # to seconds
                count += 1
        dr = np.diff(dr,axis=0)
        dt = np.diff(dt)
        # Calculate velocity (dr is now dr/dt)
        for i in [0,1,2]: # there must be a better way of doing this way
            dr[:,i] = dr[:,i]/dt
        dr_raw = dr.copy()
        
        # remove anamolously high ball velocities
        dr[np.abs(dr)>maxspeed]=0.0 # perhaps should be nan, but this would affect surrounding frames
        
        # Apply filters
        dr[:,0] = signal.savgol_filter(dr[:,0],window_length=window,polyorder=polyorder)
        dr[:,1] = signal.savgol_filter(dr[:,1],window_length=window,polyorder=polyorder)
        dr[:,2] = signal.savgol_filter(dr[:,2],window_length=window,polyorder=polyorder)
        frames[istart].ball_vx = 0.
        frames[istart].ball_vy = 0.
        frames[istart].ball_vz = 0.
        frames[istart].ball_speed_filter2 = 0
        frames[istart].ball_speed_filter3 = 0
        
        # add ball velocity to frame data
        for i in mask[1:]:
            if frames[istart+i].ball:
                v = dr[i-1,:]
                frames[istart+i].ball_vx = v[0]
                frames[istart+i].ball_vy = v[1]
                frames[istart+i].ball_vz = v[2]
                frames[istart+i].ball_speed_filter2 = np.sqrt(v[0]**2+v[1]**2)
                frames[istart+i].ball_speed_filter3 = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
        x.append(dr)
        xraw.append(dr_raw)
    dr = np.vstack(tuple(x))
    dr_raw = np.vstack(tuple(xraw))

    return frames, dr, dt, dr_raw


def estimate_player_velocities(team1_players, team0_players, match,
                               _filter='Savitzky-Golay', window=7, polyorder=1,
                               maxspeed=14, precision=3, include_acceleration=True):
    """ estimate all player velocities in vy an vy

    Arguments:
        team1_players {dict} --
        team0_players {dict} --
        match {tracab match} --

    Keyword Arguments:
        _filter {str} -- type of filter (default: {'Savitzky-Golay'})
        window {int} -- window length for filter (default: {7})
        polyorder {int} -- order for filter, 1 is linear (default: {1})
        maxspeed {int} -- maximum speed in m/s (default: {14})
    """
    # Frame of interest is in the center of the window
    all_players = list(team0_players.items()) + list(team1_players.items())
    for (num, player) in all_players:
        # initialize
        nframes = len(player.frame_targets)
        dr = np.zeros((nframes, 2), dtype=float)
        dt = np.zeros(nframes, dtype=float)

        for i, frame in enumerate(player.frame_targets):
            dr[i, :] = np.array([frame.pos_x, frame.pos_y])/100. # to m
            dt[i] = player.frame_timestamps[i] * 60  # to seconds # FIX THIS! 

            # Add time stamps to tracab_target
            player.frame_targets[i].timestamp = player.frame_timestamps[i] * 60
            frame.vx, frame.vy, frame.v_filter = np.nan, np.nan, np.nan

        dr = np.diff(dr, axis=0)
        dt = np.diff(dt)
        for i in [0, 1]:
            dr[:, i] = dr[:, i]/dt

        # remove anamolously high  velocities
        dr[np.abs(dr) > maxspeed] = 0.0  # perhaps should be nan, but this would affect surrounding frames

        # Apply filters to get velocities
        dr[:, 0] = signal.savgol_filter(dr[:, 0], window_length=window, polyorder=polyorder)
        dr[:, 1] = signal.savgol_filter(dr[:, 1], window_length=window, polyorder=polyorder)

        # get accelerations
        if include_acceleration:
            a = np.zeros((nframes - 1, 2), dtype=float)
            dt_mode = scipy.stats.mode(dt).mode
            a[:, 0] = signal.savgol_filter(dr[:, 0], window_length=window, polyorder=polyorder, deriv=1, delta=dt_mode)
            a[:, 1] = signal.savgol_filter(dr[:, 1], window_length=window, polyorder=polyorder, deriv=1, delta=dt_mode)

        # Put information back into frames
        for i, frame in enumerate(player.frame_targets[1:]):
            frame.vx = round(dr[i, 0], precision)
            frame.vy = round(dr[i, 1], precision)
            frame.v_filter = round(np.sqrt(frame.vx**2 + frame.vy**2), precision)
            if include_acceleration:
                frame.ax = round(a[i, 0], precision)
                frame.ay = round(a[i, 1], precision)
                frame.a_filter = round(np.sqrt(frame.ax**2 + frame.ay**2), precision)


def estimate_com_frames(frames_tb, match_tb, team1exclude, team0exclude):
    """ center of mass velocity for each team

    could help to compare to th

    Arguments:
        frames_tb {[type]} -- [description]
        match_tb {[type]} -- [description]
        team1exclude {[type]} -- [description]
        team0exclude {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    for frame in frames_tb:
        # home team
        players = frame.team1_players
        frame.team1_x = 0.0
        frame.team1_y = 0.0
        frame.team1_vx = 0.
        frame.team1_vy = 0.
        nv = 0
        pids = players.keys()
        pids = [pid for pid in pids if pids not in team1exclude]
        for pids in pids:
            frame.team1_x += players[pids].pos_x
            frame.team1_y += players[pids].pos_y
            frame.team1_vx += players[pids].vx
            frame.team1_vy += players[pids].vy
            nv += 1
        if nv>0:
            frame.team1_x = frame.team1_x / float(nv)
            frame.team1_y = frame.team1_y / float(nv)
            frame.team1_vx = frame.team1_vx / float(nv)
            frame.team1_vy = frame.team1_vy / float(nv)
        else:
            frame.team1_x = np.nan
            frame.team1_y = np.nan
            frame.team1_vx = np.nan
            frame.team1_vy = np.nan
        # away team
        players = frame.team0_players
        frame.team0_x = 0.0
        frame.team0_y = 0.0
        frame.team0_vx = 0.
        frame.team0_vy = 0.
        nv = 0
        pids = players.keys()
        pids = [pid for pid in pids if pids not in team0exclude]
        for pids in pids:
            frame.team0_x += players[pids].pos_x
            frame.team0_y += players[pids].pos_y
            frame.team0_vx += players[pids].vx
            frame.team0_vy += players[pids].vy
            nv += 1
        if nv>0:
            frame.team0_x = frame.team0_x / float(nv)
            frame.team0_y = frame.team0_y / float(nv)
            frame.team0_vx = frame.team0_vx / float(nv)
            frame.team0_vy = frame.team0_vy / float(nv)
        else:
            frame.team0_x = np.nan
            frame.team0_y = np.nan
            frame.team0_vx = np.nan
            frame.team0_vy = np.nan

    return frames_tb
    