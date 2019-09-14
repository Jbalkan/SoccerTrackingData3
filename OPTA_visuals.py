# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:08:14 2019

@author: laurieshaw
"""

import OPTA as opta
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
import Tracking_Visuals as vis

def get_all_matches():
    # get all matches in a range (for testing)
    ids = np.arange(984455,984635)
    fpath = "/Users/laurieshaw/Documents/Football/Data/TrackingData/Tracab/SuperLiga/All/"
    matches = []
    for i in ids:
        try:
            match_OPTA = opta.read_OPTA_f7(fpath,str(i)) 
            match_OPTA = opta.read_OPTA_f24(fpath,str(i),match_OPTA)
            matches.append(match_OPTA)
            print i
        except:
            print "error in %d" % (i)
    return matches
    
def xG_calibration_plots(matches, caley_type = [1,2,3,4,5,6], bins = np.linspace(0,1,11)):
    # poal probability in bins of xG
    shots = []
    for match in matches:
        shots =  shots + [e for e in match.hometeam.events if e.is_shot] + [e for e in match.awayteam.events if e.is_shot]
    xG_caley = [shot for shot in shots if shot.caley_types in caley_type]
    xG_caley2 = [shot for shot in shots if shot.caley_types2 in caley_type]
    xG_total_caley = np.sum([s.expG_caley for s in xG_caley])
    xG_total_caley2 = np.sum([s.expG_caley2 for s in xG_caley2])
    G_total = np.sum([s.is_goal for s in xG_caley])
    G_total2 = np.sum([s.is_goal for s in xG_caley2])
    print "xG_caley1: %1.2f, xG: %1.2f, goals = %d, %d" % (xG_total_caley,xG_total_caley2,G_total,G_total2)
    binned_xG_caley = np.zeros( shape = len(bins)-1)
    binned_xG_caley2 = np.zeros( shape = len(bins)-1)
    pgoal_caley = np.zeros( shape = len(bins)-1)
    pgoal_caley2 = np.zeros( shape = len(bins)-1)
    n_caley = np.zeros( shape = len(bins)-1)
    for i,b in enumerate(bins[:-1]):
        binned_xG_caley[i] = np.mean( [s.expG_caley for s in xG_caley if s.expG_caley>=b and s.expG_caley<=bins[i+1]])
        binned_xG_caley2[i] = np.mean( [s.expG_caley2 for s in xG_caley2 if s.expG_caley2>=b and s.expG_caley2<=bins[i+1]] )
        pgoal_caley[i] = np.mean( [s.is_goal for s in xG_caley if s.expG_caley>=b and s.expG_caley<=bins[i+1]] )
        pgoal_caley2[i] = np.mean( [s.is_goal for s in xG_caley2 if s.expG_caley2>=b and s.expG_caley2<=bins[i+1]] )
        n_caley[i] = float( len( [s for s in xG_caley if s.expG_caley>=b and s.expG_caley<=bins[i+1] ] ) )
    fig,ax = plt.subplots()
    std = np.sqrt(pgoal_caley2*(1-pgoal_caley2)/n_caley)
    
    ax.errorbar(binned_xG_caley2,pgoal_caley2,2*std)
    ax.plot(binned_xG_caley,pgoal_caley,'r')
    ax.plot([0,1],[0,1],'k--')
    
    
def plot_all_shots(match_OPTA,plotly=False):
    symbols = lambda x: 'd' if x=='Goal' else 'o'
    fig,ax = vis.plot_pitch(match_OPTA)
    homeshots = [e for e in match_OPTA.hometeam.events if e.is_shot]
    awayshots = [e for e in match_OPTA.awayteam.events if e.is_shot]
    xfact = match_OPTA.fPitchXSizeMeters*100
    yfact = match_OPTA.fPitchYSizeMeters*100
    descriptors = {}
    count = 0
    for shot in homeshots:
        descriptors[str(count)] = shot.shot_descriptor
        ax.plot(shot.x*xfact,shot.y*yfact,'r'+symbols(shot.description),alpha=0.6,markersize=20*np.sqrt(shot.expG_caley),label=count)
        count += 1
    for shot in awayshots:
        descriptors[str(count)] = shot.shot_descriptor
        ax.plot(-1*shot.x*xfact,-1*shot.y*yfact,'b'+symbols(shot.description),alpha=0.6,markersize=20*np.sqrt(shot.expG_caley),label=count)
        count += 1
    home_xG = np.sum([s.expG_caley2 for s in homeshots])
    away_xG = np.sum([s.expG_caley2 for s in awayshots])
    match_string = '%s %d (%1.1f) vs (%1.1f) %d %s' % (match_OPTA.hometeam.teamname,match_OPTA.homegoals,home_xG,away_xG,match_OPTA.awaygoals,match_OPTA.awayteam.teamname)
    #ax.text(-75*len(match_string),match_OPTA.fPitchYSizeMeters*50+500,match_string,fontsize=20)
    
    if plotly:
        plt.title( match_string, fontsize=20, y=0.95 )
        plotly_fig = tls.mpl_to_plotly( fig )
        for d in plotly_fig.data:
            if d['name'] in descriptors.keys():
                d['text'] = descriptors[d['name']]
                d['hoverinfo'] = 'text'
            else:
                d['name'] = ""
                d['hoverinfo'] = 'name'
                
        plotly_fig['layout']['titlefont'].update({'color':'black', 'size':20, 'family':'monospace'})
        plotly_fig['layout']['xaxis'].update({'ticks':'','showticklabels':False})
        plotly_fig['layout']['yaxis'].update({'ticks':'','showticklabels':False})
        #url = py.plot(plotly_fig, filename = 'Aalborg-match-analysis')
        return plotly_fig
    else:
        plt.title( match_string, fontsize=16, y=1.0 )
        return fig,ax
        
def make_expG_timeline(match_OPTA):
    fig,ax = plt.subplots()
    homeshots = [e for e in match_OPTA.hometeam.events if e.is_shot]
    awayshots = [e for e in match_OPTA.awayteam.events if e.is_shot]
    homeshots = sorted(homeshots,key=lambda x: x.time)
    awayshots = sorted(awayshots,key=lambda x: x.time)
    homeshot_expG = [s.expG_caley2 for s in homeshots] 
    awayshot_expG = [s.expG_caley2 for s in awayshots]  
    home_xG = np.sum( homeshot_expG )
    away_xG = np.sum( awayshot_expG)
    homeshot_time = [s.time for s in homeshots]
    awayshot_time=  [s.time for s in awayshots]
    homegoal_times = [s.time for s in homeshots if s.is_goal]
    awaygoal_times = [s.time for s in awayshots if s.is_goal]
    homegoal_xG_totals = [np.sum([s.expG_caley2 for s in homeshots if s.time<=t]) for t in homegoal_times]
    awaygoal_xG_totals = [np.sum([s.expG_caley2 for s in awayshots if s.time<=t]) for t in awaygoal_times]
    # for step chart
    homeshot_time = [0] + homeshot_time + [match_OPTA.total_match_time]
    awayshot_time = [0] + awayshot_time + [match_OPTA.total_match_time]
    homeshot_expG = [0]+ homeshot_expG + [0.0]
    awayshot_expG = [0]+ awayshot_expG + [0.0]
    plt.step( homeshot_time, np.cumsum(homeshot_expG), 'r',where='post')
    plt.step( awayshot_time, np.cumsum(awayshot_expG), 'b',where='post')
    ax.set_xticks(np.arange(0,100,10))
    ax.xaxis.grid(alpha=0.5)
    ax.yaxis.grid(alpha=0.5)
    match_string = '%s %d (%1.1f) vs (%1.1f) %d %s' % (match_OPTA.hometeam.teamname,match_OPTA.homegoals,home_xG,away_xG,match_OPTA.awaygoals,match_OPTA.awayteam.teamname)
    fig.suptitle(match_string,fontsize=16,y=0.95)
    ax.set_xlim(0,match_OPTA.total_match_time)
    ymax = np.ceil([max(home_xG,away_xG)])
    ax.set_ylim(0,ymax)
    ax.text(match_OPTA.total_match_time+1,home_xG,match_OPTA.hometeam.teamname,color='r')
    ax.text(match_OPTA.total_match_time+1,away_xG,match_OPTA.awayteam.teamname,color='b')
    ax.set_ylabel('Cummulative xG',fontsize=14)
    ax.set_xlabel('Match time',fontsize=14)
    print homegoal_times, homegoal_xG_totals
    ax.plot(homegoal_times,homegoal_xG_totals,'ro')
    ax.plot(awaygoal_times,awaygoal_xG_totals,'bo')
    # add goal scorers
    home_scorers = ['Goal (' + s.player_name +')' for s in homeshots if s.is_goal]
    away_scorers = ['Goal (' + s.player_name +')' for s in awayshots if s.is_goal]
    if len(home_scorers)>0:
        [ax.text(t-0.2,x,n,horizontalalignment='right',fontsize=8,color='r') for t,x,n in zip(homegoal_times,homegoal_xG_totals,home_scorers)]
    if len(away_scorers)>0:
        [ax.text(t-0.2,x,n,horizontalalignment='right',fontsize=8,color='b') for t,x,n in zip(awaygoal_times,awaygoal_xG_totals,away_scorers)]
        
        
def plot_defensive_actions(team,all_matches,include_tackles=True,include_intercept=True):
    match_OPTA = all_matches[0]
    # tackles won and retained posession
    team_events = []
    for match in all_matches:
        if match.hometeam.teamname==team:
            team_events += match.hometeam.events
        elif match.awayteam.teamname==team:
            team_events += match.awayteam.events
    tackle_won = [e for e in team_events if e.type_id == 7 and e.outcome==1 and 167 not in e.qual_id_list]
    # interceptions
    intercepts = [e for e in team_events if e.type_id == 8]
    fig,ax = vis.plot_pitch(all_matches[0])
    xfact = match_OPTA.fPitchXSizeMeters*100
    yfact = match_OPTA.fPitchYSizeMeters*100
    count = 0
    # tackles
    if include_tackles:
        for tackle in tackle_won:
            ax.plot(tackle.x*xfact,tackle.y*yfact,'rd',alpha=0.6,markersize=10,label=count)
            count +=1
    if include_intercept:
        for intercept in intercepts:
            ax.plot(intercept.x*xfact,intercept.y*yfact,'rs',alpha=0.6,markersize=10,label=count)
            count +=1
    #match_string = '%s %d (%1.1f) vs (%1.1f) %d %s' % (match_OPTA.hometeam.teamname,match_OPTA.homegoals,home_xG,away_xG,match_OPTA.awaygoals,match_OPTA.awayteam.teamname)
    #plt.title( match_string, fontsize=16, y=1.0 )
            
def Generate_Tracab_Chance_Videos(match_OPTA, match_tb, frames_tb):
    t_init_buf = 0.5
    t_end_buf = 0.1
    match_end = frames_tb[-1].timestamp
    all_shots = [e for e in match_OPTA.hometeam.events if e.is_shot] + [e for e in match_OPTA.awayteam.events if e.is_shot]
    for shot in all_shots:
        print shot
        tstart = (shot.period_id, max(0.0,shot.time-45*(shot.period_id-1) - t_init_buf))
        tend = (shot.period_id, min(match_end,shot.time-45*(shot.period_id-1) + t_end_buf))
        frames_in_segment = vis.get_frames_between_timestamps(frames_tb,match_tb,tstart,tend)
        vis.save_match_clip(frames_in_segment,match_tb,fpath=match_OPTA.fpath+'/chances/',fname=shot.shot_id,include_player_velocities=False,description=shot.shot_descriptor)
   