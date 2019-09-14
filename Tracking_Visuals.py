# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:20:39 2019

Module to visualise player positions and velocities on the pitch, including movies

@author: laurieshaw
"""

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import graham_scan as gs

def plot_pitch(match,fig=None,ax=None,lw=2,ps=20,no_center_spot=False):
    # Plots a pitch. Pitch dimensions are often defined in yards, so need to convert distance units
    # set up plot
    if fig is None:
        fig,ax = plt.subplots(figsize=(12,8))
    lw=lw
    lc = 'k'
    # ALL DIMENSIONS IN cm
    xborder = 300
    yborder = 300
    cm_per_yard = 91.44
    half_pitch_width = match.fPitchYSizeMeters*100/2. # in cm
    half_pitch_length = match.fPitchXSizeMeters*100/2.
    signs = [-1,1]
    # yards to cm
    goal_line_width = 8*cm_per_yard
    box_width = 20*cm_per_yard
    box_length = 6*cm_per_yard
    area_width = 44*cm_per_yard
    area_length = 18*cm_per_yard
    penalty_spot = 12*cm_per_yard
    corner_radius = 1*cm_per_yard
    D_length = 8*cm_per_yard
    D_radius = 10*cm_per_yard
    D_pos = 12*cm_per_yard
    centre_circle_radius = 10*cm_per_yard
    # half way line # center circle
    ax.plot([0,0],[-half_pitch_width,half_pitch_width],lc,linewidth=lw)
    if not no_center_spot:
        ax.scatter(0.0,0.0,marker='o',facecolor=lc,linewidth=0,s=ps)
    y = np.linspace(-1,1,50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x,y,lc,linewidth=lw)
    ax.plot(-x,y,lc,linewidth=lw)
    for s in signs:
        # plot pitch boundary
        ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=lw)
        ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=lw)
        # goal posts & line
        ax.plot( [s*half_pitch_length,s*half_pitch_length],[-goal_line_width/2.,goal_line_width/2.],lc+'s',markersize=6*ps/20.,linewidth=lw)
        # 6 yard box
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[box_width/2.,box_width/2.],lc,linewidth=lw)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[-box_width/2.,-box_width/2.],lc,linewidth=lw)
        ax.plot([s*half_pitch_length-s*box_length,s*half_pitch_length-s*box_length],[-box_width/2.,box_width/2.],lc,linewidth=lw)
        # penalty area
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[area_width/2.,area_width/2.],lc,linewidth=lw)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[-area_width/2.,-area_width/2.],lc,linewidth=lw)
        ax.plot([s*half_pitch_length-s*area_length,s*half_pitch_length-s*area_length],[-area_width/2.,area_width/2.],lc,linewidth=lw)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot,0.0,marker='o',facecolor=lc,linewidth=0,s=ps)
        # corner flags
        y = np.linspace(0,1,50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x,-half_pitch_width+y,lc,linewidth=lw)
        ax.plot(s*half_pitch_length-s*x,half_pitch_width-y,lc,linewidth=lw)
        # draw the D
        y = np.linspace(-1,1,50)*D_length
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x,y,lc,linewidth=lw)
        
    #ax.patch.set_facecolor("mediumspringgreen")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    xmax = match.fPitchXSizeMeters*100/2. + xborder
    ymax = match.fPitchYSizeMeters*100/2. + yborder
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    return fig,ax

def get_frames_between_timestamps(frames,match,tstart,tend):
    # returns frames between two timestamps
    # tstart, tend are tuples (half [1,2], frame timestamp)
    if match.provider == 'Tracab':
        ihalf = match.period_attributes[tstart[0]]['iStart']
    elif match.provider == 'SecondSpectrum': 
        ihalf = match.period_framenum[tstart[0]-1]
    timestamps = np.array( [ f.timestamp for f in frames[ihalf:] ] )
    fstart = np.argmax( timestamps>=tstart[1] ) + ihalf
    fend = np.argmax( timestamps>=tend[1] )  + ihalf
    return frames[fstart:fend]

def find_framenum_at_timestamp(frames,match,half,timestamp):
    # returns first frame after a given point in time
    # timestamp is a tuple (half [1,2], frame timestamp)
    if match.provider == 'Tracab':
        ihalf = match.period_attributes[half]['iStart']
    elif match.provider == 'SecondSpectrum': 
        ihalf = match.period_framenum[half-1]
    timestamps = np.array( [ f.timestamp for f in frames[ihalf:] ] )
    framenum = np.argmax( timestamps>=timestamp ) + ihalf
    return framenum

def plot_frame(frame,match,alpha=1.0,include_player_velocities=True,include_ball_velocities=True, figax=None):
    # plots positions (and velocities) of players 
    if figax is None:
        fig,ax = plot_pitch(match)
    else:
        (fig,ax) = figax
    team1 = np.zeros((14,4))
    team0 = np.zeros((14,4))
    for i,j in enumerate( frame.team1_jersey_nums_in_frame ):
        team1[i,0] = frame.team1_players[j].pos_x
        team1[i,1] = frame.team1_players[j].pos_y
        team1[i,2] = frame.team1_players[j].vx/2.
        team1[i,3] = frame.team1_players[j].vy/2.
    team1 = team1[:i+1,:]
    for i,j in enumerate( frame.team0_jersey_nums_in_frame ):
        team0[i,0] = frame.team0_players[j].pos_x
        team0[i,1] = frame.team0_players[j].pos_y
        team0[i,2] = frame.team0_players[j].vx/2. # Divide by 2 for aesthetic purposes only (otherwise quiver is too long)
        team0[i,3] = frame.team0_players[j].vy/2. # Divide by 2 for aesthetic purposes only (otherwise quiver is too long)
    team0 = team0[:i+1,:]
    pts1, = ax.plot( team1[:,0], team1[:,1], 'ro',linewidth=0,markeredgewidth=0,markersize=10,alpha=alpha)
    pts2, = ax.plot( team0[:,0], team0[:,1], 'bo',linewidth=0,markeredgewidth=0,markersize=10,alpha=alpha)
    ax.quiver( team1[:,0], team1[:,1], team1[:,2], team1[:,3],color='r',width=0.002,headlength=5,headwidth=3,scale=60)
    ax.quiver( team0[:,0], team0[:,1], team0[:,2], team0[:,3],color='b',width=0.002,headlength=5,headwidth=3,scale=60)
    if frame.ball:
        ax.plot( frame.ball_pos_x, frame.ball_pos_y, marker='o',markerfacecolor='k',markeredgewidth=0,markersize=6*max(1,np.sqrt(frame.ball_pos_z/150.)), alpha=alpha)
        if include_ball_velocities:
            x = np.array([0,frame.ball_vx*100]) + frame.ball_pos_x
            y = np.array([0,frame.ball_vy*100]) + frame.ball_pos_y
            ax.plot( x, y, 'k', alpha=alpha)
    
    ax.text(-150,match.fPitchYSizeMeters*50+40,frame.min+':'+ frame.sec.zfill(2), fontsize=14  )
    txt_offset = 30    
    for i, n in enumerate( frame.team1_jersey_nums_in_frame ): 
        ax.annotate(n, (team1[i,0]+txt_offset,team1[i,1]+txt_offset))
    for i, n in enumerate( frame.team0_jersey_nums_in_frame ): 
        ax.annotate(n, (team0[i,0]+txt_offset,team0[i,1]+txt_offset))
    return fig,ax

def plot_frames(frames,match,pause=0.0,include_player_velocities=True,include_ball_velocities=False, skip=1,alpha=1.0):
    # sequentially plots 
    fig,ax = plot_pitch(match)
    points_on = False
    ball_on = False
    team1 = np.zeros((14,4))
    team0 = np.zeros((14,4))
    for frame in frames[::skip]:
        if points_on: # we need to remove points from previous frame!
            pts1.remove()
            pts2.remove()
            pts3.remove()
            pts6.remove()
            pts7.remove()
        if ball_on:
            pts4.remove()
            if include_ball_velocities:
                pts5.remove()
        for i,j in enumerate( frame.team1_jersey_nums_in_frame ):
            team1[i,0] = frame.team1_players[j].pos_x
            team1[i,1] = frame.team1_players[j].pos_y
            team1[i,2] = frame.team1_players[j].vx/4.
            team1[i,3] = frame.team1_players[j].vy/4.
        team1 = team1[:i+1,:]
        for i,j in enumerate( frame.team0_jersey_nums_in_frame ):
            team0[i,0] = frame.team0_players[j].pos_x
            team0[i,1] = frame.team0_players[j].pos_y
            team0[i,2] = frame.team0_players[j].vx/4. # Divide by 2 for aesthetic purposes only (otherwise quiver is too long)
            team0[i,3] = frame.team0_players[j].vy/4. # Divide by 2 for aesthetic purposes only (otherwise quiver is too long)
        team0 = team0[:i+1,:]
        pts1, = ax.plot( team1[:,0], team1[:,1], 'ro',linewidth=0,markeredgewidth=0,markersize=10)
        pts2, = ax.plot( team0[:,0], team0[:,1], 'bo',linewidth=0,markeredgewidth=0,markersize=10)
        pts6 = ax.quiver( team1[:,0], team1[:,1], team1[:,2], team1[:,3],color='r',width=0.002,headlength=5,headwidth=3,scale=60)
        pts7 = ax.quiver( team0[:,0], team0[:,1], team0[:,2], team0[:,3],color='b',width=0.002,headlength=5,headwidth=3,scale=60)
        if frame.ball and frame.ball_status=='Alive':
            ball_on=True
            pts4, = ax.plot( frame.ball_pos_x, frame.ball_pos_y, marker='o',markerfacecolor='k',markeredgewidth=0,markersize=6*max(1,np.sqrt(frame.ball_pos_z/150.)))
            if include_ball_velocities:
                x = np.array([0,frame.ball_vx*100]) + frame.ball_pos_x
                y = np.array([0,frame.ball_vy*100]) + frame.ball_pos_y
                pts5, = ax.plot( x, y, 'k', alpha=alpha,linewidth=0.5)
        else:
            ball_on = False
        pts3 = ax.text(-150,match.fPitchYSizeMeters*50+40,frame.min+':'+ frame.sec.zfill(2), fontsize=14 )
        if pause>0:
            plt.pause(pause)
        if not points_on:
            points_on = True
        
def save_match_clip(frames,match,fpath,fname='clip_test',include_player_velocities=True,hcol='r',acol='b',team1_exclude=[],team0_exclude=[],include_hulls=False,speed_fact=1.0,description=None):
    # saves a movie of frames with filename 'fname' in directory 'fpath'
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='test clip!')
    writer = FFMpegWriter(fps=match.iFrameRateFps*speed_fact, metadata=metadata)
    fig,ax = plot_pitch(match)
    fig.suptitle(fname,fontsize=16,y=0.95)
    if description is not None:
        ax.text( -1*match.fPitchXSizeMeters/2. *100, -1*(match.fPitchYSizeMeters/2.+10)*100, description, fontsize=12, alpha=0.7)
    points_on = False
    in_play_flag_on = False
    team1 = np.zeros((14,4))
    team0 = np.zeros((14,4))
    fname = fpath + fname + '.mp4'
    with writer.saving(fig, fname, 100):
        for frame in frames:
            if points_on:
                pts1.remove()
                pts2.remove()
                pts3.remove()
                pts4.remove()
                if include_player_velocities:
                    pts6.remove()
                    pts7.remove()
                if include_hulls:
                    pts8.remove()
                    pts9.remove()
            c = 0
            for i,j in enumerate( frame.team1_jersey_nums_in_frame ):
                if j not in team1_exclude:
                    team1[c,0] = frame.team1_players[j].pos_x
                    team1[c,1] = frame.team1_players[j].pos_y
                    team1[c,2] = frame.team1_players[j].vx/2.
                    team1[c,3] = frame.team1_players[j].vy/2.
                    c += 1
            team1 = team1[:c,:]
            c = 0
            for i,j in enumerate( frame.team0_jersey_nums_in_frame ):
                if j not in team0_exclude:
                    team0[c,0] = frame.team0_players[j].pos_x
                    team0[c,1] = frame.team0_players[j].pos_y
                    team0[c,2] = frame.team0_players[j].vx/2.
                    team0[c,3] = frame.team0_players[j].vy/2.
                    c += 1
            team0 = team0[:c,:]
            pts1, = ax.plot( team1[:,0], team1[:,1], hcol+'o',linewidth=0,markeredgewidth=0,markersize=10)
            pts2, = ax.plot( team0[:,0], team0[:,1], acol+'o',linewidth=0,markeredgewidth=0,markersize=10)
            if include_player_velocities:       
                pts6 = ax.quiver( team1[:,0], team1[:,1], team1[:,2], team1[:,3],color=hcol,width=0.002,headlength=5,headwidth=3,scale=80)
                pts7 = ax.quiver( team0[:,0], team0[:,1], team0[:,2], team0[:,3],color=acol,width=0.002,headlength=5,headwidth=3,scale=80)
            if include_hulls:
                hull1 = gs.graham_scan(team1[:,:2],enclose=True)
                hull0 = gs.graham_scan(team0[:,:2],enclose=True)
                pts8, = ax.fill(hull1[:,0], hull1[:,1], hcol, alpha=0.3)
                pts9, = ax.fill(hull0[:,0], hull0[:,1], acol, alpha=0.3)
            if frame.ball:
                pts3, = ax.plot( frame.ball_pos_x, frame.ball_pos_y, marker='o',markerfacecolor='k',markeredgewidth=0,markersize=6*max(1,np.sqrt(frame.ball_pos_z/150.)))
                if frame.ball_status!='Alive' and not in_play_flag_on:
                    pts10 = ax.text(match.fPitchXSizeMeters*50-1500,match.fPitchYSizeMeters*50-250,'Play stopped', fontsize=14, color='r' )
                    in_play_flag_on = True
                elif frame.ball_status=='Alive' and in_play_flag_on:
                    pts10.remove()
                    in_play_flag_on = False
            else:
                pts3, = ax.plot( 0, 0, marker='x',markerfacecolor='k',markeredgewidth=0,markersize=6) 
            pts4 = ax.text(-150,match.fPitchYSizeMeters*50+40,str((frame.period-1)*45+int(frame.min))+':'+ frame.sec.zfill(2), fontsize=14 )
            if not points_on:
                points_on = True
            writer.grab_frame()
    plt.close('all')
            
            
def plot_player_timeseries(frames,match,team,jerseynum):
    # extracts positions over frames for a single player identified by jerseynum
    nframes = len(frames)
    px = np.zeros(nframes)
    py = np.zeros(nframes)
    sp = np.zeros(nframes)
    timestamp = np.zeros(nframes)
    for i,frame in enumerate(frames):
        offset = 0 if frame.period == 1 else 45
        timestamp[i] = frame.timestamp + offset
        if team==1:
            if jerseynum in frame.team1_jersey_nums_in_frame:
                px[i] = frame.team1_players[jerseynum].pos_x
                py[i] = frame.team1_players[jerseynum].pos_y
                sp[i] = frame.team1_players[jerseynum].speed
            else:
                px[i] = None
                py[i] = None
                sp[i] = None
        elif team==0:
            if jerseynum in frame.team0_jersey_nums_in_frame:
                px[i] = frame.team0_players[jerseynum].pos_x
                py[i] = frame.team0_players[jerseynum].pos_y
                sp[i] = frame.team0_players[jerseynum].speed
            else:
                px[i] = None
                py[i] = None
                sp[i] = None
    return timestamp,px,py,sp
    