# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:06:54 2019 

Module for reading Tracab data and constructing frame and player objects, 
and calculating possessions

@author: laurieshaw, modified by Jeff Balkanski

"""

import os
from bs4 import BeautifulSoup
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import Tracking_Velocities as vel
import xml.etree.ElementTree as ET


def get_filename_dict_EPL():
    """ Get a dictionary of file names for EPL legauge games
    
    Returns:
        dict -- filename dict for EPL league
    """
    raw_file_names = "/Users/laurieshaw/Documents/Football/Data/TrackingData/Tracab/RawData/raw_file_names.txt"
    fname_dict = {}
    with open(raw_file_names, "r") as rf:
        for l in rf:
            l = l.split(":")
            # print l[0],l[1]
            fname_dict[l[0]] = l[1].replace('\n', '')

    return fname_dict


def get_tracabdata_paths(fpath, fname, league='EPL'):
    """ Get paths of tracabdata, modified by Jeff to work with first example
    
    Arguments:
        fpath {str} -- filepath of data
        fname {str} -- file name
    
    Keyword Arguments:
        league {str} -- league name (default: {'EPL'})
    
    Returns:
        [type] -- fmetadata
        [type] -- fdata
    """
    if league == 'EPL':
        fmetadata = fpath+'TracabMetadata'+fname
        fdata = fpath+'TracabData'+fname

    elif league == 'DSL':
        fmetadata = os.path.join(fpath, fname + "_metadata.xml")
        fdata = os.path.join(fpath, fname + '.dat')

    # # previous
    # elif league =='DSL':
    #     fmetadata = fpath+fname+"/"+fname+"_metadata.xml"
    #     fdata = fpath+fname+"/"+fname+".dat"

    return fmetadata, fdata


def read_tracab_match(fmetadata):
    """ Create tracab object out of XML metadata (not used)

    Arguments:
        fmetadata {str} -- file path of meta data

    Returns:
        tracab_match -- match object of corresponding paths
    """
    # get meta data
    f = open(fmetadata, "r")
    metadata = f.read()
    soup = BeautifulSoup(metadata, 'xml') # deal with xml markup (BS might be overkill here)
    match_attributes = soup.match.attrs
    period_attributes = {}

    # navigate through the file to extract the info that we need
    for p in soup.find_all('period'):
        if p.attrs['iEndFrame']>p.attrs['iStartFrame']:
            period_attributes[int(p.attrs['iId'])] = p.attrs

    match = tracab_match(match_attributes, period_attributes)

    return match


def read_tracab_match_xml(fmetadata):
    """ Read tracab match xml file (not used here)

    Arguments:
        fmetadata {str} -- file path of meta data

    Returns:
        tracab_match -- match object of corresponding paths
    """
    # get meta data
    tree = ET.parse(fmetadata)
    root = tree.getroot()
    match_attributes = root.getchildren()[0].attrib
    period_attributes = {}
    temp = root.getchildren()[0].getchildren()

    # navigate through the file to extract the info that we need
    for t in temp:
        if int(t.attrib['iEndFrame']) > int( t.attrib['iStartFrame']):
            period_attributes[int(t.attrib['iId'])] = t.attrib

    match = tracab_match(match_attributes, period_attributes)

    return match


def read_tracab_match_data(league, fpath, fname,
                           team1_exclude=None, team0_exclude=None,
                           during_match_only=True, verbose=False):
    """ Main function to read match data
        Create a list of frame objects and a tracab_match

    Arguments:
        league {str} -- league name
        fpath {str} -- [description]
        fname {str} -- [description]

    Keyword Arguments:
        team1_exclude {[type]} -- [description] (default: {None})
        team0_exclude {[type]} -- [description] (default: {None})
        during_match_only {bool} -- [description] (default: {True})
        verbose {bool} -- show print statements (default: {False})

    Returns:
        list -- frames
        tracab_match -- match
        list -- team1_players
        list -- team0_players
    """
    # get match metadata
    fmetadata, fdata = get_tracabdata_paths(fpath, fname, league)
    if verbose:
        print("Reading match metadata")
    match = read_tracab_match_xml(fmetadata)

    # read in tracking data
    frames = []
    if verbose:
        print("Reading match tracking data")
    with open(fdata, "r") as fp:
        # go through line by line and break down data in individual players and the ball
        for f in fp:
            # each line is a single frame
            chunks = f.split(':')[:-1]  # last element is carriage return \n
            if len(chunks) > 3:
                raise Exception('More than 3 chunks in line of data: %s', chunks)

            # create a frame obj
            frameid = int(chunks[0])
            frame = tracab_frame(frameid)

            # Get players
            targets = chunks[1].split(';')
            assert(targets[-1] == '')
            for target in targets[:-1]:
                target = target.split(',')
                team = int(target[0])
                if team in [1, 0, 3]:
                    frame.add_frame_target(target)

            # Is this never the case?
            if len(chunks) > 2:
                frame.add_frame_ball(chunks[2].split(';')[0].split(','))

            frames.append(frame)

    # sort the frames by frameid (should be sorted anyway, just to make sure)
    frames = sorted(frames, key=lambda x: x.frameid)

    # timestamp frames
    if verbose:
        print("Timestamping frames")
    frames, match = timestamp_frames(frames, match)

    # run some basic checks
    check_frames(frames)
    
    # remove pre-match, post-match and half-time frames
    if during_match_only:  
        frames = frames[match.period_attributes[1]['iStart']:match.period_attributes[1]['iEnd']+1] + frames[match.period_attributes[2]['iStart']:match.period_attributes[2]['iEnd']+1]
        match.period_attributes[1]['iEnd'] = match.period_attributes[1]['iEnd']-match.period_attributes[1]['iStart']
        match.period_attributes[1]['iStart'] = 0
        match.period_attributes[2]['iEnd'] = match.period_attributes[2]['iEnd'] - match.period_attributes[2]['iStart'] + match.period_attributes[1]['iEnd']
        match.period_attributes[2]['iStart'] = match.period_attributes[1]['iEnd']+1

    # identify which way each team is shooting
    set_parity(frames, match)

    # get player objects
    if verbose:
        print("Measuring velocities")
    team1_players, team0_players = get_players(frames)
    if team1_exclude is None or team0_exclude is None:
        team1_exclude, team0_exclude = get_goalkeeper_numbers(frames)

    match.team1_exclude = team1_exclude
    match.team0_exclude = team0_exclude

    # player metrics
    vel.estimate_player_velocities(team1_players, team0_players, match, window=7, polyorder=1, maxspeed=14)
    vel.estimate_player_accelerations(team1_players, team0_players, match)

    #  ball and team com velocity  
    vel.estimate_ball_velocities(frames, match, window=5, polyorder=3, maxspeed=40)
    vel.estimate_com_frames(frames, match, team1_exclude, team0_exclude)

    return frames, match, team1_players, team0_players


def set_parity(frames, match):
    # determines the direction in which the home team are shooting
    # 1: right->left, -1: left->right
    match.period_parity = {}
    team1_x = np.mean([frames[0].team1_players[pid].pos_x for pid in frames[0].team1_jersey_nums_in_frame])
    if team1_x > 0:
        match.period_parity[1] = 1
        match.period_parity[2] = -1
    else:
        match.period_parity[1] = -1
        match.period_parity[2] = 1

    return match


def check_frames(frames):
    frameids = [frame.frameid for frame in frames]
    missing = set(frameids).difference(set(range(min(frameids),max(frameids)+1)))
    nduplicates = len(frameids)-len(np.unique(frameids))
    if len(missing)>0:
        print ("Check Fail: Missing frames")
    if nduplicates>0:
        print ("Check Fail: Duplicate frames found")


def get_goalkeeper_numbers(frames, verbose=True):
    ids = frames[0].team1_jersey_nums_in_frame
    x = []
    for i in ids:
        x.append(frames[0].team1_players[i].pos_x)
    team1_exclude = [ids[np.argmax(np.abs(x))]]
    # check to see if there is a goalkeeper sub
    if team1_exclude[0] not in frames[-1].team1_jersey_nums_in_frame:
        team1_exclude = team1_exclude + find_GK_substitution(frames,1,team1_exclude[0])
    ids = frames[0].team0_jersey_nums_in_frame
    x = []
    for i in ids:
        x.append( frames[0].team0_players[i].pos_x )
    team0_exclude = [ ids[np.argmax( np.abs( x ) )] ]
    if team0_exclude[0] not in frames[-1].team0_jersey_nums_in_frame:
        team0_exclude = team0_exclude + find_GK_substitution(frames,0,team0_exclude[0])
    if verbose:
        print ("home goalkeeper(s): ", team1_exclude)
        print ("away goalkeeper(s): ", team0_exclude)

    return team1_exclude, team0_exclude


def find_GK_substitution(frames,team,first_gk_id):
    """ 
    find goal keeper id at each frame 
    throw an exception 
    
    Arguments:
        frames {[type]} -- [description]
        team {[type]} -- [description]
        first_gk_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    gk_id = first_gk_id
    new_gk = []
    plast = []
    for frame in frames[::25]: # look every second
        if frame.ball_status == 'Alive': # ball should be back in play
            if team==1:
                pnums = frame.team1_jersey_nums_in_frame
            else:
                pnums = frame.team0_jersey_nums_in_frame
            if gk_id not in pnums:
                sub = set(pnums) - set(plast)
                print (sub)
                if len(sub)!=1:
                    print( "No or more than one substitute")
                    assert False
                else:
                    new_gk.append( int(list(sub)[0]) )
                    gk_id = new_gk[-1]
            plast = pnums
    if len(new_gk) != 1:
        print( "goalkeeper sub problem", new_gk)
    return new_gk


def get_players(frames):
    """ Get all players appearing in frames argument
    
    Arguments:
        frames {lst[frames]} -- list of frames to identify players
    
    Returns:
        (dict, dict) -- dictionaries of player objects
    """
    # first get all players that appear in at least one frames
    team1_jerseys = set([])  # home team
    team0_jerseys = set([])  # away team
    team1_players = {}
    team0_players = {}
    for frame in frames:
        team1_jerseys.update(frame.team1_jersey_nums_in_frame)
        team0_jerseys.update(frame.team0_jersey_nums_in_frame)

    # get all jerseys in frames
    for j1 in team1_jerseys:
        team1_players[j1] = tracab_player(j1, 1)
    for j0 in team0_jerseys:
        team0_players[j0] = tracab_player(j0, 0)

    for frame in frames:
        for j in team1_jerseys:
            if j in frame.team1_jersey_nums_in_frame:
                team1_players[j].add_frame(frame.team1_players[j], frame.frameid, frame.timestamp)
            else:  # player is not in this frame
                team1_players[j].add_null_frame(frame.frameid, frame.timestamp)
        for j in team0_jerseys:
            if j in frame.team0_jersey_nums_in_frame:
                team0_players[j].add_frame(frame.team0_players[j], frame.frameid, frame.timestamp)
            else:  # player is not in this frame
                team0_players[j].add_null_frame(frame.frameid, frame.timestamp)

    return team1_players, team0_players


def timestamp_frames(frames, match):
    """ add information to frames such as period and timestamp
    
    Arguments:
        frames {list} -- list of tracab_frame
        match {traca_match} -- match object
    
    Returns:
        lst -- frames
        tracab match -- match
    """
    # Frames must be sorted into ascending frameid first
    frame_period = 1/float(match.iFrameRateFps)

    # ASSUMES NO INJURY TIME
    match.period_attributes[1]['iStart'] = None
    match.period_attributes[2]['iStart'] = None
    match.period_attributes[1]['iEnd'] = None
    match.period_attributes[2]['iEnd'] = None
    for i, frame in enumerate(frames):
        if frame.frameid < match.period_attributes[1]['iStartFrame']:
            frame.period = 0  # pre match
            frame.timestamp = -1

        elif frame.frameid > match.period_attributes[2]['iEndFrame']:
            frame.period = 4  # post match
            frame.timestamp = -1

        elif frame.frameid <= match.period_attributes[1]['iEndFrame']:
            frame.period = 1  # first half
            frame.timestamp = (frame.frameid-match.period_attributes[1]['iStartFrame']) * frame_period / 60.
            frame.min = str(int(frame.timestamp))
            frame.sec = "%1.2f" % (round((frame.timestamp-int(frame.timestamp)) * 60., 3))
            if match.period_attributes[1]['iStart'] is None:
                match.period_attributes[1]['iStart'] = i
            if frame.frameid == match.period_attributes[1]['iEndFrame']:
                match.period_attributes[1]['iEnd'] = i
                match.period_attributes[1]['iEndTime'] = frame.timestamp

        elif frame.frameid >= match.period_attributes[2]['iStartFrame']:
            frame.period = 2  # second half
            frame.timestamp = (frame.frameid-match.period_attributes[2]['iStartFrame']) * frame_period / 60.
            frame.min = str(int(frame.timestamp))
            frame.sec = "%1.2f" % (round((frame.timestamp-int(frame.timestamp)) * 60., 3))
            if match.period_attributes[2]['iStart'] is None:
                match.period_attributes[2]['iStart'] = i
            if frame.frameid == match.period_attributes[2]['iEndFrame']:
                match.period_attributes[2]['iEnd'] = i
                match.period_attributes[2]['iEndTime'] = frame.timestamp
        else:
            frame.period = 3  # half time
            frame.timestamp = -1

    return frames, match


def get_tracab_posessions(frames, match, min_pos_length=0):
    """ get posesssions from list of frames
    
    Arguments:
        frames {lst} -- list of frames to get posessions
        match {tracab match} -- [description]
    
    Keyword Arguments:
        min_pos_length {int} -- minimum posession length (default: {0})
    
    Returns:
        lst -- list of posessions
    """
    posessions = []
    for p in match.period_attributes.keys():
        newpos = True
        period = match.period_attributes[p]
        for i in np.arange(period['iStart'],period['iEnd']+1):
            if newpos:
                if frames[i].ball_status == 'Alive':
                    istart = i
                    newpos = False
            else:
                if frames[i].ball_status != 'Alive' or frames[i].ball_team != frames[istart].ball_team:
                    posessions.append( tracab_possesion(frames[istart],frames[i-1],istart,match))
                    newpos = True
        if not newpos:
            posessions.append( tracab_possesion(frames[istart],frames[i], istart,match) )
    posessions = [pos for pos in posessions if pos.pos_duration>min_pos_length]
    for pos in posessions:
        pos.set_possession_type(frames, match)

    return posessions


def make_posession_plot(posessions):
    """ Make a posession plot
    
    Arguments:
        posessions {list} -- [description]
    """
    home = [p.pos_duration for p in posessions if p.team=='H']
    away = [p.pos_duration for p in posessions if p.team=='A']
    fig,ax = plt.subplots()
    xbins = np.arange(0,61,4)
    print(np.mean(home),np.median(home))
    print(np.sum(home)/60., np.sum(away)/60., np.sum(home)/60.+np.sum(away)/60.)
    ax.hist(away,xbins,color='w',edgecolor='blue', linewidth=1.2,alpha=1,linestyle='dashed',label='away')
    ax.hist(home,xbins,color='r',alpha=0.3,label='home')
    ax.set_ylim(0,75)
    ax.plot([np.mean(home),np.mean(home)],[60,75],'r--')
    ax.plot([np.median(home),np.median(home)],[60,75],'r:')
    ax.plot([np.mean(away),np.mean(away)],[60,75],'b--')
    ax.plot([np.median(away),np.median(away)],[60,75],'b:')
    ax.set_xlabel('Possession duration (seconds)')
    ax.legend(framealpha=0.2,frameon=False,fontsize=10,numpoints=1)
    

#############################################################
#                                                           #
#                       Classes                             #
#                                                           #
#############################################################


class tracab_match(object):
    """ Tracab Match class
    
    Arguments:
        object {[type]} -- match_attributes
        object {[type]} -- period_attributes
    """
    def __init__(self, match_attributes, period_attributes):
        self.provider = 'Tracab'
        self.match_attributes = match_attributes
        self.date = dt.datetime.strptime(match_attributes['dtDate'], '%Y-%m-%d %H:%M:%S')
        self.fPitchXSizeMeters = float(match_attributes['fPitchXSizeMeters'])
        self.fPitchYSizeMeters = float(match_attributes['fPitchYSizeMeters'])
        self.fTrackingAreaXSizeMeters = float(match_attributes['fTrackingAreaXSizeMeters'])
        self.fTrackingAreaYSizeMeters = float(match_attributes['fTrackingAreaYSizeMeters'])
        self.iFrameRateFps = int(match_attributes['iFrameRateFps'])
        for pk in period_attributes.keys():
            for ak in period_attributes[pk].keys():
                period_attributes[pk][ak] = int(period_attributes[pk][ak])
        self.period_attributes = period_attributes
        

class tracab_frame(object):
    """ Tracab Frame object
    
    Arguments:
        object {int} -- frameid

    """
    def __init__(self, frameid):
        self.provider = 'Tracab'
        self.frameid = frameid
        self.team1_players = {}
        self.team0_players = {}
        self.team1_jersey_nums_in_frame = []
        self.team0_jersey_nums_in_frame = []
        self.ball = False
        self.referee = None
        
    def add_frame_target(self, target_raw):
        # add a player to the frame
        team = int( target_raw[0] )
        sys_target_ID = int( target_raw[1] )
        jersey_num = int( target_raw[2] )
        pos_x = float( target_raw[3] )  # useful to make positions a float
        pos_y = float( target_raw[4] )
        speed = float( target_raw[5] )
        if team == 1:
            self.team1_players[jersey_num] = tracab_target(team,sys_target_ID,jersey_num,pos_x,pos_y,speed) 
            self.team1_jersey_nums_in_frame.append( jersey_num )
        elif team == 0:
            self.team0_players[jersey_num] = tracab_target(team,sys_target_ID,jersey_num,pos_x,pos_y,speed) 
            self.team0_jersey_nums_in_frame.append( jersey_num )
        else:
            self.referee = tracab_target(team,sys_target_ID,jersey_num,pos_x,pos_y,speed) 
    
    def add_frame_ball(self,ball_raw):
        self.ball = True
        self.ball_pos_x = float( ball_raw[0] )
        self.ball_pos_y = float( ball_raw[1] )
        self.ball_pos_z = float( ball_raw[2] )
        self.ball_speed = float( ball_raw[3] ) / 100. # ball speed is in cm/s?
        self.ball_team = ball_raw[4]
        self.ball_status = ball_raw[5]
        if len(ball_raw)==7:
            self.ball_contact_info =ball_raw[6]
        else:
            self.ball_contact_info = None
        
    def __repr__(self):
        nplayers = len(self.team1_jersey_nums_in_frame)+len(self.team1_jersey_nums_in_frame)
        nrefs = 0 if self.referee is None else 1
        s = 'Frame id: %d, nplayers: %d, nrefs: %d, nballs: %d' % (self.frameid, nplayers, nrefs, self.ball*1)
        return s


class tracab_target(object):
    """  defines position of an individual target 'player' in a frame
    
    Arguments:
        object {[type]} -- team
        object {[type]} -- sys_target_ID
        object {[type]} -- jersey_num
        object {[type]} -- pos_x
        object {[type]} -- pos_y
        object {[type]} -- speed, given by the data source

    """
    def __init__(self, team, sys_target_ID, jersey_num, pos_x, pos_y, speed):
        self.team = team
        self.sys_target_ID = sys_target_ID
        self.jersey_num = jersey_num
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.speed = speed


class tracab_player(object):
    """ tracab player obj
    
    Arguments:
        object {int} -- jersey_num
        object {int} -- teamID
    """
    # contains trajectory of a single player over the entire match
    def __init__(self, jersey_num, teamID):
        self.jersey_num = jersey_num
        self.teamID = teamID
        self.frame_targets = []
        self.frame_timestamps = []
        self.frameids = []

    def add_frame(self, target, frameid, timestamp):
        self.frame_targets.append( target )
        self.frame_timestamps.append( timestamp )
        self.frameids.append( frameid )
        
    def add_null_frame(self,frameid,timestamp):
        # player is not on the pitch (specifically, is not in a frame)
        self.frame_timestamps.append( timestamp )
        self.frameids.append( frameid )
        self.frame_targets.append( tracab_target(self.teamID, None, self.jersey_num, 0.0, 0.0, 0.0 ))


class tracab_possesion(object):
    """ Tracab Possesion class
    
    Arguments:
        object {[type]} -- fstart
        object {[type]} -- fend
        object {[type]} -- fstart_num
        object {tracab match} -- match

    """
    # a single period of continuous posession for a team
    def __init__(self, fstart, fend, fstart_num, match):
        self.period = fstart.period
        self.pos_start_fid = fstart.frameid
        self.pos_end_fid = fend.frameid
        self.pos_start_time = fstart.timestamp
        self.pos_end_time = fend.timestamp
        self.pos_start_matchtime = self.pos_start_time + (self.period-1)*match.period_attributes[1]['iEndTime']
        self.pos_end_matchtime = self.pos_end_time + (self.period-1)*match.period_attributes[1]['iEndTime']
        self.pos_duration = 60*(self.pos_end_time - self.pos_start_time)
        self.pos_Nframes = self.pos_end_fid - self.pos_start_fid
        self.team = fstart.ball_team
        self.pos_start_fnum = fstart_num
        self.pos_end_fnum = fstart_num+self.pos_Nframes
        self.prev_pos_type = None
        # check if team changes during the possesion (suggesions a data error in ball status)
        self.bad_possesion = ( set( fstart.team1_jersey_nums_in_frame ) != set( fend.team1_jersey_nums_in_frame ) ) or ( set( fstart.team0_jersey_nums_in_frame ) != set( fend.team0_jersey_nums_in_frame ) )
        if self.bad_possesion:
            print( "bad posession: period %d, %1.2f to %1.2f" % (self.period, self.pos_start_time,self.pos_end_time))
        assert self.team==fend.ball_team
        assert fstart.period==fend.period
        
    def set_possession_type(self,frames_tb,match_tb):
        frames = frames_tb[self.pos_start_fnum:self.pos_end_fnum+1]
        if self.team=='H': # team shooting from right->left
            # ball-based metric
            #x = np.median( [f.ball_pos_x for f in frames] ) * match_tb.period_parity[self.period]
            # team-based metric
            x = np.median( [f.team1_x for f in frames] ) * match_tb.period_parity[self.period]
        else: # team shooting from left->right
            # ball-based metric
            #x = np.median( [f.ball_pos_x for f in frames] ) * -1*match_tb.period_parity[self.period]
            x = np.median( [f.team0_x for f in frames] ) * -1*match_tb.period_parity[self.period]
        if x < 0:
            self.pos_type = 'A'
        elif x > match_tb.fPitchXSizeMeters*100/4.:
            self.pos_type = 'D'
        else:
            self.pos_type = 'N' # neutral
        self.set_possession_start(frames_tb)
        
    def set_possession_start(self,frames_tb):
        if self.pos_start_fnum==0 or frames_tb[self.pos_start_fnum-1].ball_status=='Dead':
            self.pos_start_type = 'DeadBall'
        else:
            self.pos_start_type = 'WinPosession'
        
        
    def __repr__(self):
        s = 'Team %s, Period %d, Type %s, start = %1.2f, end = %1.2f, length = %1.2f' % (self.team,self.period,self.pos_type,self.pos_start_time,self.pos_end_time,self.pos_duration)
        return s
