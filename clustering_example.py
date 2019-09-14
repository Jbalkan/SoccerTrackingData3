# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:15:55 2019

@author: laurieshaw
"""

import Tracab as tracab
import Tracking_Formation as form
import Tracking_Visuals as vis
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import cPickle as pickle

fpath='/Path/To/Directory/of/Tracking/Data/' # path to directory of Tracab data

match_ids = range(984455,984634+1)
missing = [984474,984488,984616] # data problems in these three matches, so remove them
[match_ids.remove(m) for m in missing]

all_formations = []
formation_stats = []



for match in match_ids[0:1]:
    print match
    fname = str(match)
    # read frames, match meta data, and data for individual players
    frames_tb, match_tb, team1_players, team0_players = tracab.read_tracab_match_data('DSL',fpath,fname,verbose=False)

    # get goalkeeper ids
    team1_exclude = match_tb.team1_exclude
    team0_exclude = match_tb.team0_exclude

    # get all possessions
    posessions = tracab.get_tracab_posessions(frames_tb,match_tb,min_pos_length=1)
    
    # get home and away possessions, remove possessions less than 5 seconds (unless they start with a dead ball) and those in the defensive quarter of the field
    posH = [p for p in posessions if p.team=='H' and p.pos_type in ['A','N'] and (p.pos_duration>5 or p.pos_start_type=='DeadBall') and not p.bad_possesion]
    posA = [p for p in posessions if p.team=='A' and p.pos_type in ['A','N'] and (p.pos_duration>5 or p.pos_start_type=='DeadBall') and not p.bad_possesion]
    
    min_pos_length = 60. # aggregated possession windows must be at least this many seconds
    max_pos_length = 120. # and at most this many
    subsample = 10 # downsample at this rate
    
    # measure offensive and defensive formations for home and away possessions
    A,D = form.calc_formation_by_period(max_pos_length,min_pos_length,posH,frames_tb,match_tb,subsample,team1_exclude,team0_exclude,plotfig=False,deleteLattices=True)
    A2,D2 = form.calc_formation_by_period(max_pos_length,min_pos_length,posA,frames_tb,match_tb,subsample,team1_exclude,team0_exclude,plotfig=False,deleteLattices=True)
    # record formation measures for match
    all_formations = all_formations + A + D + A2 + D2
    # keep a record of how many we have
    formation_stats.append((match,len(A),len(D),len(A2),len(D2)))

    # some memory management
    del frames_tb
    del team1_players
    del team0_players
    
print "total number of formations: %d" % (len(all_formations))
Amatrix = np.zeros((len(all_formations),len(all_formations)))
Smatrix = np.zeros((len(all_formations),len(all_formations)))

for i in range(len(all_formations)):
    if np.mod(i,100)==0:
        print i
    for j in range(i,len(all_formations)):
        if i==j:
            Amatrix[i,j] = 0.
        else:
            cost,p1,pm,s = form.Hungarian_Cost(all_formations[i],all_formations[j],match_tb,metric='WM',squeeze=[0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5],plot=False)
            Amatrix[i,j] = cost
            Smatrix[i,j] = s
            Amatrix[j,i] = Amatrix[i,j]
            Smatrix[j,i] = Smatrix[i,j]

print "finished calculating matrix. saving."

#plt.matshow(Amatrix)
#plt.show()        

fname = "formation_measures.pkl"
results = [formation_stats,all_formations,Amatrix,Smatrix]
#results = [formation_stats,Amatrix]

with open(fname, 'wb') as output:
    pickle.dump(results,output,pickle.HIGHEST_PROTOCOL)



