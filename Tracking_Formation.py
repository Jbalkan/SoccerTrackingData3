# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:06:44 2019

@author: laurieshaw
"""

import Tracab as tracab
import Tracking_Visuals as vis
import numpy as np
import graham_scan as gs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment
import scipy.cluster.hierarchy as sch
from scipy.linalg import sqrtm
import copy as copy
import itertools
import data_utils as utils
import cPickle as pickle
import scipy.stats as stats


def calc_formation_by_period(period_length,min_period_length,posessions,frames_tb,match_tb,subsample,excludeH,excludeA,plotfig=False,deleteLattices=False):
    # aggregates individual posessions into windoes of at least min_period_length and at most period_length
    # then calculates attackoing (team in possession) and defensive (team out of possession) formations in each possession window
    # posessions is a list of possessions
    # excludeH/excludeA are player ids to exclude (the goalkeepers)
    # subsample reduces framerate to speed up calculation
    # note all posessions **must** be the same team
    team = posessions[0].team # the team that has possession in the first possession of the list (Home or away)
    # group posessions into chunks
    period_posessions = [] # each element corresponds to a list of frames for the aggregated possessions
    count = 0.0  # this is a counter that tracks how much playing time we have in each aggregated possession window
    frames = []
    # set of player ids that appear in the first frame of the match
    pids_H = set( frames_tb[0].team1_jersey_nums_in_frame )
    pids_A = set( frames_tb[0].team0_jersey_nums_in_frame )
    for pos in posessions:
        assert pos.team==team # make sure that we haven't switched team
        # has there been a substituion?
        sub = pids_H != set( frames_tb[pos.pos_start_fnum].team1_jersey_nums_in_frame ) or pids_A != set( frames_tb[pos.pos_start_fnum].team0_jersey_nums_in_frame )
        if sub: # there was a substituion
            #print "sub", count, frames_tb[pos.pos_start_fnum].timestamp, frames_tb[pos.pos_end_fnum].timestamp
            period_posessions.append(frames) # end possession window
            frames = frames_tb[pos.pos_start_fnum:pos.pos_end_fnum+1] # start new possession window
            count = pos.pos_duration
            pids_H = set( frames_tb[pos.pos_start_fnum].team1_jersey_nums_in_frame )
            pids_A = set( frames_tb[pos.pos_start_fnum].team0_jersey_nums_in_frame )
        else:
            frames = frames + frames_tb[pos.pos_start_fnum:pos.pos_end_fnum+1]
            if count >= period_length: # if we have enough time in aggregated possessions, start a new possession window
                #print count
                period_posessions.append(frames)
                frames = []
                count = 0
            else:
                count += pos.pos_duration
    #  now we have a list of aggregated possession windows. Remove those that are too short (because of substitutions)
    period_posessions = [p for p in period_posessions if len(p)>min_period_length*float(match_tb.iFrameRateFps)]
    attacking_formations = []
    defensive_formations = []
    # now measure the attacking and defensive formations in each possession window
    for period_posession in period_posessions:
        form_Att = formation(team) # 'team' here is technically the team in possession, so the same in both cases
        form_Def = formation(team)
        period_start = period_posession[0].period
        for frame in period_posession[::subsample]: # subsampling to speed this up
            if team=='H':
                attacking_players = frame.team1_players
                defending_players = frame.team0_players
                parity = match_tb.period_parity[frame.period] # deals with left-right play direction
                excludeAtt = excludeH # these are the player ids to exclude (goalkeepers)
                excludeDef = excludeA
            else:
                attacking_players = frame.team0_players
                defending_players = frame.team1_players
                parity = match_tb.period_parity[frame.period]*-1
                excludeAtt = excludeA
                excludeDef = excludeH
            form_Att.add_latice( lattice('A',parity,frame.timestamp,attacking_players,excludeAtt ) ) # always make this an attacking 'A' formation, so that the team shoots from right->left
            form_Def.add_latice( lattice('A',parity*-1,frame.timestamp,defending_players,excludeDef) )
        form_Def.calc_average_lattice(match_tb)
        form_Att.calc_average_lattice(match_tb)
        offsets = calc_offset_between_formations(form_Def,form_Att,calc='offside',parity=parity)
        if plotfig:
            fig,ax = vis.plot_pitch(match_tb)
            fig1,ax1 = vis.plot_pitch(match_tb)
            form_Def.plot_formation(factor=100,figax=(fig,ax),lt='bo')
            form_Att.plot_formation(factor=100,offsets=offsets,figax=(fig1,ax1),lt='ro')
        if deleteLattices: # reduce memory usage
            form_Def.delete_lattices()
            form_Att.delete_lattices()
        attacking_formations.append(form_Att)
        defensive_formations.append(form_Def)
    return attacking_formations,defensive_formations
    

def calc_posession_lattice(posessions,frames_tb,match_tb,subsample):
    # parity=1, home team is attacking right->left
    frames = []
    fig,ax = vis.plot_pitch(match_tb)
    fig1,ax1 = vis.plot_pitch(match_tb)
    # note all posessions **must** be the same team
    team = posessions[0].team
    for pos in posessions:
        assert pos.team==team
        frames = frames + frames_tb[pos.pos_start_fnum:pos.pos_end_fnum+1]
    form_Att = formation(team)
    form_Def = formation(team)
    for frame in frames[::subsample]:
        if posessions[0].team=='H':
            attacking_players = frame.team1_players
            defending_players = frame.team0_players
            parity = match_tb.period_parity[frame.period]
        else:
            attacking_players = frame.team0_players
            defending_players = frame.team1_players
            parity = match_tb.period_parity[frame.period]*-1
        form_Att.add_latice( lattice('A',parity,frame.timestamp,attacking_players,[1,31]) )
        form_Def.add_latice( lattice('D',parity*-1,frame.period,frame.timestamp,defending_players,[1,31]) )
    print (form_Att.pids)
    print (form_Def.pids
    form_Def.calc_average_lattice)(match_tb)
    form_Att.calc_average_lattice(match_tb)
    offsets = calc_offset_between_formations(form_Def,form_Att,calc='offside',parity=parity)
    form_Def.plot_formation(factor=100,figax=(fig,ax),lt='bo')
    form_Att.plot_formation(factor=100,offsets=offsets,figax=(fig1,ax1),lt='ro')
    return form_Def,form_Att

    
def calc_offset_between_formations(form_Def,form_Att,calc,parity=1):
    if calc=='com':
        assert len(form_Def.lattices) == len(form_Att.lattices)
        nlattices = len(form_Def.lattices)
        xoffset = 0.0
        yoffset = 0.0
        for i in range(nlattices):
            xoffset = form_Def.lattices[i].xcom - form_Att.lattices[i].xcom
            yoffset = form_Def.lattices[i].ycom - form_Att.lattices[i].ycom
        xoffset = xoffset/float(nlattices)
        yoffset = yoffset/float(nlattices)
    if calc=='offside':
        if (form_Att.team=='H' and parity==1) or (form_Att.team=='A' and parity==2):
            xoffset = form_Def.bounds[0] - form_Att.bounds[0]
            yoffset = 0.
        else:
            xoffset = form_Def.bounds[1] - form_Att.bounds[1]
            yoffset = 0.    
    return xoffset,yoffset


def make_dendogram(Amatrix,formation_stats,t=80,method='ward',dendogram=True):
    # calculate dendogram from matrix of distances between formation observations
    # returns a list that indicates the cluster for each formation observation based on the horizontal distance threshold 't'
    d = sch.distance.squareform(Amatrix)
    L = sch.linkage( d, method=method)
    c = sch.cophenet(L)
    rho_p = stats.pearsonr(c,d)[0]
    rho_s = stats.spearmanr(c,d).correlation
    print ("Cophenetic distance = %1.2f (spearman), %1.2f (pearson)" % (rho_s,rho_p))
    if dendogram:
        fig,ax = plt.subplots(figsize=(25, 10))
        #dn = sch.dendrogram(L)
        dn = fancy_dendrogram(L,truncate_mode='lastp',p=200,leaf_rotation=90.,leaf_font_size=12.,show_contracted=True,annotate_above=10)
    # group clusters
    fcl = sch.fcluster(L,t=t,criterion='distance',depth=2)
    # now map formations to clusters
    ctypes = []
    tmat = ['A','D','A','D']
    count = 0
    for f in formation_stats:
        #teams = (f[0][0:3],f[0][3:],f[0][3:],f[0][0:3])
        teams = ('H','A','A','H')
        for i in [1,2,3,4]:
            for j in range(f[i]):
                ctypes.append( (count,f[0],teams[i-1],tmat[i-1], fcl[count]) )
                count += 1
        ctypes = sorted(ctypes, key = lambda x: x[4] )
    return ctypes


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = sch.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

    
def generate_formation_from_sub_cluster(ctypes,all_formations,clusternums,Amatrix, match_tb, method='ward', squeeze=[0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]):
    '''
    rather than crudely averaging together formations in the same sub-cluster, we should build them hierarchically. 'cluster nums' is the clusters identified in the dendogram
    plus cut-off that determine which formation observations will be combined. Amatrix is the distance matrix between the observations. Squeeze contains the values of the
    compactness parameter that we marginalize over.
    '''
    # first need to get the formation ids in the sub-cluster. Also remove formations with less than 10 outfield players
    ctypes = [c for c in ctypes if c[4] in clusternums and len( all_formations[c[0]].pids )==10]
    formation_nums = [c[0] for c in ctypes]
    #print len(formation_nums)
    cluster_formations = [all_formations[n] for n in formation_nums]
    nformations = len(formation_nums)
    # now generate the distance submatrix for this cluster(s)
    Asubmatrix = np.zeros((nformations,nformations))
    for i in range(nformations):
        for j in range( nformations):
            Asubmatrix[i,j] = Amatrix[ formation_nums[i], formation_nums[j] ]
    # convert distances to a 1-dy array
    d = sch.distance.squareform(Asubmatrix)
    # do the clustering
    L = sch.linkage(d, method=method)
    #dn = fancy_dendrogram(L,truncate_mode='lastp',p=100,leaf_rotation=90.,leaf_font_size=12.,show_contracted=True,annotate_above=10)
    # now work way up tree
    scale = match_tb.fPitchYSizeMeters*2./3. # set a cluster 'scale' width. Will try to use this to ensure averaged formation observations don't shrink or grow too much
    for cluster_formation in cluster_formations: # initialize some values
        cluster_formation.n_in_cluster = 1
        cluster_formation.nodes_cluster = {}
        for pid in cluster_formation.pids:
            x = cluster_formation.nodes[pid].x
            y = cluster_formation.nodes[pid].y
            cluster_formation.nodes_cluster[pid] = np.array( [x,y] )
        
    for link in L:
        # now work our way up the dendogram, combining formations as we go
        f1 = cluster_formations[int(link[0])] # first formation/cluster
        f2 = cluster_formations[int(link[1])] # second formation/cluster
        f1.width = max([f1.nodes[pid].y for pid in f1.pids]) - min([f1.nodes[pid].y for pid in f1.pids])
        f2.width = max([f2.nodes[pid].y for pid in f2.pids]) - min([f2.nodes[pid].y for pid in f2.pids])
        if np.abs(f1.width-scale) > np.abs(f2.width-scale):
            # switch ordering to try to maintain constant scale size
            f1 = cluster_formations[int(link[1])]
            f2 = cluster_formations[int(link[0])]
        # calculate Wasserstein distance again, marginalising over the 'squeeze' parameters
        c, pids1, pids2, s = Hungarian_Cost(f1,f2,match_tb,metric='WM',plot=False,squeeze=squeeze)
        # create a new formation cluster from the combined formations
        clustered_formation = formation('C')
        clustered_formation.is_cluster_template = False
        clustered_formation.nodes ={}
        clustered_formation.nodes_cluster = {}
        clustered_formation.formation_distributions = {}
        clustered_formation.n_in_cluster = f1.n_in_cluster + f2.n_in_cluster
        w1 = f1.n_in_cluster/float(clustered_formation.n_in_cluster)
        w2 = f2.n_in_cluster/float(clustered_formation.n_in_cluster)
        # quick check that the number of formation observations in the combined cluster is what it is supposed to be.
        assert clustered_formation.n_in_cluster==link[3]
        for i,pid in enumerate(pids1):
            # average positions
            xm = w1*f1.formation_distributions[pid][0][0] + w2*f2.formation_distributions[pids2[i]][0][0]*s
            ym = w1*f1.formation_distributions[pid][0][1] + w2*f2.formation_distributions[pids2[i]][0][1]*s
            cov = f1.formation_distributions[pid][1]*w1 + f2.formation_distributions[pids2[i]][1]*w2*s*s
            clustered_formation.formation_distributions[i] = [np.array([xm,ym]), cov]
            clustered_formation.nodes[i] = node(i, xm, ym )
            clustered_formation.nodes_cluster[i] = np.vstack( ( f1.nodes_cluster[pid], f2.nodes_cluster[pids2[i]]*s ) )
        clustered_formation.pids = clustered_formation.nodes.keys()
        cluster_formations.append(clustered_formation)
    # get the final formation at the top of the tree
    cluster_formation = cluster_formations[-1]
    # model distributions of positions for each player in formation as bivariate normal
    cluster_formation.calc_player_distributions_within_cluster_formation()
    cluster_formation.is_cluster_template = True
    cluster_formation.ctypes = ctypes # save these for later analysis
    return cluster_formation

def get_superliga_clusters(Amatrix,Smatrix,all_formations,formation_stats,match_tb):
    # generates clusters for latest presentation
    ctypes =  make_dendogram( Amatrix,formation_stats,t=80,method='ward')
    clusternums = [[1],[2],[3],[4,5],[6,7,8],[9],[10],[11],[12],[13,14],[15,16],[17,18,19],[20],[21],[22],[23],[24],[25],[26],[27,28]]
    nclusters = len(clusternums)   
    print nclusters
    clustered_formations = []
    for cluster in clusternums:
        print cluster
        clustered_formations.append( generate_formation_from_sub_cluster(ctypes, all_formations, cluster, Amatrix, Smatrix, match_tb) )
    assert len(clustered_formations)==20
    return clustered_formations

def save_clustered_formations(clustered_formations):
    fname = "/Users/laurieshaw/Documents/Football/Data/TrackingData/Tracab/clustered_formations_WMs.pkl"
    with open(fname, 'wb') as output:
        pickle.dump(clustered_formations, output, pickle.HIGHEST_PROTOCOL)    
        
def load_clustered_formations():
    fname = "/Users/laurieshaw/Documents/Football/Data/TrackingData/Tracab/clustered_formations_WMs.pkl"
    with open(fname, 'rb') as input:
        clustered_formations = pickle.load(input) 
    return clustered_formations    

    
def Hungarian_Cost(F1,F2,match_tb,metric='L2',squeeze=[1.],plot=False):
    # calculates cost of player assignments between any two roles (L1 or L2 norm) in each formation
    # this process tells us how to match a player in one formation observation to a player in another to minimize the overal formation comparison cost
    # uses the Kuhn-Munkres algorithm
    pids1 = np.array(F1.pids)
    pids2 = np.array(F2.pids)
    n1 = len(pids1)
    n2 = len(pids2)
    if len(squeeze)>1: # marginalise over the squeeze paramter
        cost = []
        for s in squeeze:
            C = np.zeros(shape=(n1,n2))
            for i in range(n1):
                pid1 = pids1[i]
                for j in np.arange(n2):
                    pid2 = pids2[j]
                    C[i,j] = Cost_Metric(F1,F2,pid1,pid2,metric=metric,s=s)
            row_ind, col_ind =  linear_sum_assignment(C)
            cost.append((s,C[row_ind, col_ind].sum(),col_ind))
        cost = sorted(cost,key=lambda x: x[1])
        col_ind_min = cost[0][2]
        cost_min = cost[0][1]
        squeeze_min = cost[0][0]
        #print squeeze_min
        pmatch = pids2[col_ind_min]
    else: # just use a single squeeze parameter. Obviously can simplify this.
        squeeze_min = squeeze[0]
        C = np.zeros(shape=(n1,n2))
        for i in range(n1):
            pid1 = pids1[i]
            for j in range(n2):
                pid2 = pids2[j]
                C[i,j] = Cost_Metric(F1,F2,pid1,pid2,metric=metric,s=squeeze_min)
        row_ind, col_ind =  linear_sum_assignment(C)
        cost_min = C[row_ind, col_ind].sum()
        pmatch = pids2[col_ind]
    if plot:
        fig,ax = vis.plot_pitch(match_tb)
        F1.plot_formation(factor=100,figax=(fig,ax),lt='ro')
        F2.plot_formation(factor=100,figax=(fig,ax),lt='bo')
        for i in range(n1):
            pid1 = pids1[i]
            pid2 = pmatch[i]
            ax.plot( [100*F1.nodes[pid1].x,squeeze_min*100*F2.nodes[pid2].x] , [100*F1.nodes[pid1].y,squeeze_min*100*F2.nodes[pid2].y], 'k' )
    return cost_min, pids1, pmatch, squeeze_min 

def Cost_Metric(F1,F2,pid1,pid2,metric='L2',s=1.):
    if metric=='L1':
        return np.abs( ( F1.nodes[pid1].x-s*F2.nodes[pid2].x ) ) + np.abs( ( F1.nodes[pid1].y-s*F2.nodes[pid2].y ) )
    elif metric=='L2':
        # square of Euclidean distance
        return ( F1.nodes[pid1].x-s*F2.nodes[pid2].x )**2 + ( F1.nodes[pid1].y-s*F2.nodes[pid2].y )**2
    elif metric=='L2W':
        # variance-weighted distance
        var1_x = F1.formation_distributions[pid1][1][0,0]
        var1_y = F1.formation_distributions[pid1][1][1,1]
        var2_x = F2.formation_distributions[pid2][1][0,0]*s*s
        var2_y = F2.formation_distributions[pid2][1][1,1]*s*s
        return ( F1.nodes[pid1].x-s*F2.nodes[pid2].x )**2/(var1_x+var2_x) + ( F1.nodes[pid1].y-s*F2.nodes[pid2].y )**2/(var1_y+var2_y)
    elif metric=='KL':
        # KL divergence
        mu1 = np.array([F1.nodes[pid1].x,F1.nodes[pid1].y])
        cov1 = F1.formation_distributions[pid1][1]
        mu2 = np.array([F2.nodes[pid2].x,F2.nodes[pid2].y])*s
        cov2 = F2.formation_distributions[pid2][1]*s*s
        return Sym_KL_Divergence(mu1,mu2,cov1,cov2)
    elif metric=='WM':
        # Wasserstein metric for bivariate normal
        mu1 = np.array([F1.nodes[pid1].x,F1.nodes[pid1].y])
        cov1 = F1.formation_distributions[pid1][1]
        mu2 = np.array([F2.nodes[pid2].x,F2.nodes[pid2].y])*s
        cov2 = F2.formation_distributions[pid2][1]*s*s
        return Wasserstein_Metric(mu1,mu2,cov1,cov2,method='fast')
    elif metric=='LL':
        # returns negative log-likelihood 
        cov1 = F1.formation_distributions[pid1][1]
        if F2.is_cluster_template:
            mu2 = F2.cluster_formation_distributions[pid2][0]*s
            #mu2 = np.array([F2.nodes[pid2].x,F2.nodes[pid2].y])*s
            cov2 = F2.cluster_formation_distributions[pid2][1]*s*s
        else:
            mu2 = F2.formation_distributions[pid2][0]*s
            #mu2 = np.array([F2.nodes[pid2].x,F2.nodes[pid2].y])*s
            cov2 = F2.formation_distributions[pid2][1]*s*s
        totalcov = cov1 + cov2
        invtotalcov = np.linalg.inv( totalcov )
        xdif = np.array( [F1.nodes[pid1].x-mu2[0],F1.nodes[pid1].y-mu2[1]] )
        ll =  -1*np.log(2.*np.pi)-0.5*np.log(np.linalg.det(totalcov)) - 0.5*np.dot(xdif, np.matmul(invtotalcov,xdif) )
        return -1*ll
    elif metric=='LLa':
        # returns negative log-likelihood in which only formation 1 has an associated covariance
        cov1 = F1.formation_distributions[pid1][1]
        invcov1 = np.linalg.inv( cov1 )
        xdif = np.array([F1.nodes[pid1].x-s*F2.nodes[pid2].x,F1.nodes[pid1].y-s*F2.nodes[pid2].y])
        ll =  -1*np.log(2.*np.pi)-0.5*np.log(np.linalg.det(cov1)) - 0.5*np.dot(xdif, np.matmul(invcov1,xdif) )
        return -1*ll
    elif metric=='LLb':
        # returns negative log-likelihood in which only formation 1 has an associated covariance
        if F2.is_cluster_template:
            mu2 = F2.cluster_formation_distributions[pid2][0]*s
            cov2 = F2.cluster_formation_distributions[pid2][1]*s*s
        else:
            mu2 = F2.formation_distributions[pid2][0]*s
            cov2 = F2.formation_distributions[pid2][1]*s*s
        invcov2 = np.linalg.inv( cov2 )
        xdif = np.array( [F1.nodes[pid1].x-mu2[0],F1.nodes[pid1].y-mu2[1]] )
        ll =  -1*np.log(2.*np.pi)-0.5*np.log(np.linalg.det(cov2)) - 0.5*np.dot(xdif, np.matmul(invcov2,xdif) )
        return -1*ll

def Sym_KL_Divergence(mu1,mu2,cov1,cov2):
    # Experimental metric
    d = np.linalg.matrix_rank(cov1)
    invcov2 = np.linalg.inv(cov2)
    invcov1 = np.linalg.inv(cov1)
    mudif = mu1 - mu2
    KLs = -1*d + 0.5* ( np.trace( np.matmul(invcov2,cov1) ) +  np.trace( np.matmul(invcov1,cov2) ) + np.dot(mudif, np.matmul(invcov2,mudif) ) + np.dot(mudif, np.matmul(invcov1,mudif) ) )
    return KLs

def Wasserstein_Metric(mu1,mu2,cov1,cov2,method='fast'):
    # Wasserstein Metric (see wikipedia entrance)
    if method!='fast':
        C2half = sqrtm(cov2)
        Cmult = cov1 + cov2 - 2.0*sqrtm( np.matmul(C2half,np.matmul(cov1,C2half)) )
    elif np.abs( np.sum(cov1-cov2) ) < 0.01: # for speed up
        Cmult = np.zeros((2,2))
    else:
        C2half = np.linalg.cholesky(cov2)
        Cmult = cov1 + cov2 - 2.0*np.linalg.cholesky( np.matmul(C2half,np.matmul(cov1,C2half.T)) )
    W = np.dot(mu1-mu2,mu1-mu2) + np.trace( Cmult  )
    return max(0.0,round(W,5))


def KL_Divergence(mu1,mu2,cov1,cov2):
    d = np.linalg.matrix_rank(cov1)
    invcov2 = np.linalg.inv(cov2)
    mudif = mu1 - mu2
    KL = 0.5* ( np.log(np.linalg.det(cov2)/np.linalg.det(cov1)) -d + np.trace( np.matmul(invcov2,cov1) ) +  np.dot(mudif, np.matmul(invcov2,mudif) ) ) 
    return KL
    
class formation(object):
    # defines a single observation of a formation from an aggregated possession window
    def __init__(self,team):
        self.lattices = []
        self.team=team # team in posession
        self.is_cluster_template = False # until set True
        
    def calc_player_deviations_within_formation(self,match,nexclude=0):
        # first need to find formation centre of mass within each lattice frame
        # do this by calculating com having iteratively excluded nexclude players to minimize cost
        self.create_formation_deviations()
        for lattice in self.lattices:
            if nexclude==0:
                # formation com is the same as formation com
                lattice.formation_xcom = lattice.xcom
                lattice.formation_ycom = lattice.ycom
            else:
                costs = []
                excludes = list(itertools.combinations(self.pids, nexclude))
                for ex in excludes:
                    ecom = lattice.get_com(ex)
                    c = self.lattice_formation_cost(lattice,ecom,ex)
                    costs.append((c,ecom,ex))
                costs = sorted(costs,key = lambda x: x[0])
                ecom = costs[0][1]
                lattice.formation_xcom = ecom[0]
                lattice.formation_ycom = ecom[1]
            
            # basically a correction to the com frame in the lattice
            com_shift_x = lattice.xcom - lattice.formation_xcom
            com_shift_y = lattice.ycom - lattice.formation_ycom
            ymax = match.fPitchYSizeMeters / 2.
            xmax = match.fPitchXSizeMeters / 2.
            for pid in self.pids:
                # don't let positions go off edge of field. 
                # need to think about this - is it an issue?
                dev_x = lattice.nodes[pid].x+com_shift_x
                #dev_x = np.sign(dev_x) * np.min( [xmax,np.abs(dev_x)] )
                dev_y = lattice.nodes[pid].y+com_shift_y
                #dev_y = np.sign(dev_y) * np.min( [ymax,np.abs(dev_y)] )
                # build array of all positions relative to formation CoM
                self.formation_deviations_x[pid] = np.concatenate((self.formation_deviations_x[pid],[dev_x]))
                self.formation_deviations_y[pid] = np.concatenate((self.formation_deviations_y[pid],[dev_y]))
        # finally, calc bivariate distributions for players
        self.calc_player_distributions_within_formation()      
                
    def calc_player_distributions_within_formation(self):
        self.formation_distributions = {}
        for pid in self.pids:
            mu = np.array([ np.mean(self.formation_deviations_x[pid]), np.mean(self.formation_deviations_y[pid]) ])
            cov = np.cov( self.formation_deviations_x[pid],self.formation_deviations_y[pid] )
            self.formation_distributions[pid] = (mu,cov)
            
    def calc_player_distributions_within_cluster_formation(self):
        self.cluster_formation_distributions = {}
        for pid in self.pids:
            mu = np.mean( self.nodes_cluster[pid],axis=0 )
            cov = np.cov( self.nodes_cluster[pid].T)
            self.cluster_formation_distributions[pid] = (mu,cov)
                
    def create_formation_deviations(self):
        self.formation_deviations_x = {}
        self.formation_deviations_y = {}
        for pid in self.pids:
            self.formation_deviations_x[pid] = np.array([])
            self.formation_deviations_y[pid] = np.array([])
            
    def lattice_formation_cost(self,lattice,ecom,pid_exclude):
        dr2 = 0
        frame_dif_x = lattice.xcom - ecom[0]
        frame_dif_y = lattice.ycom - ecom[1]
        for pid in self.pids:
            if pid not in pid_exclude:
                dx = lattice.nodes[pid].x - self.nodes[pid].x + frame_dif_x
                dy = lattice.nodes[pid].y - self.nodes[pid].y + frame_dif_y
                # this is the difference between the lattice position and the formation position for that player
                dr2 += dx*dx + dy*dy
        return dr2

    def add_latice(self,lattice):
        if len( self.lattices ) == 0:
            self.pids = set( lattice.pids )
        else:
            if set( lattice.pids ) != self.pids:
                print set( lattice.pids ), self.pids
                assert False
        self.lattices.append(lattice)
        
    def delete_lattices(self):
        self.lattices = []
    
    def assign_neighbour_position(self,pid,n_neighbours):
        for i in range(n_neighbours):
            pid_neighbour = self.nodes[pid].nearest_neighours[i][0]
            if not self.nodes[pid_neighbour].has_position(): # doesn't already have a position
                self.nodes[pid_neighbour].x = self.nodes[pid].x + self.nodes[pid].nearest_neighours[i][1]
                self.nodes[pid_neighbour].y = self.nodes[pid].y + self.nodes[pid].nearest_neighours[i][2]

    def find_nearest_neighbour_with_position(self,pid):
        count = 0
        while not self.nodes[self.nodes[pid].nearest_neighours[count][0]].has_position():
            count += 1
        pid_neighbour = self.nodes[pid].nearest_neighours[count][0]
        self.nodes[pid].x = self.nodes[pid_neighbour].x  - self.nodes[pid].nearest_neighours[count][1]
        self.nodes[pid].y = self.nodes[pid_neighbour].y  - self.nodes[pid].nearest_neighours[count][2]

    def to_com_frame(self):
        self.xcom = 0.
        self.ycom = 0
        for pid in self.pids:
            self.xcom += self.nodes[pid].x
            self.ycom += self.nodes[pid].y
        self.xcom = self.xcom/float(len(self.pids))
        self.ycom = self.ycom/float(len(self.pids))
        for pid in self.pids:
            self.nodes[pid].x = self.nodes[pid].x - self.xcom
            self.nodes[pid].y = self.nodes[pid].y - self.ycom

    def calc_lattice_average_com(self):
        self.lattice_average_com_x = 0.0
        self.lattice_average_com_y = 0.0
        nlattice = float(len(self.lattices))
        for lattice in self.lattices:
            self.lattice_average_com_x += lattice.xcom
            self.lattice_average_com_y += lattice.ycom
        self.lattice_average_com_x /= nlattice
        self.lattice_average_com_y /= nlattice

    def calc_average_lattice(self,match,n_neighbours=3,nexclude=1):
        # the key method for averaging together lattices to get a formation observation
        self.nodes = {}
        for p in self.pids:
            newnode = node(p,np.nan,np.nan)
            for n in self.pids:
                if n!=p:
                    px = []
                    py = []
                    for lattice in self.lattices:
                        #print lattice.nodes[p].neighbours.keys()
                        px.append( lattice.nodes[p].neighbours[n][1] ) # x diff
                        py.append( lattice.nodes[p].neighbours[n][2] ) # y diff
                    newnode.add_neighbour(n,np.median(px),np.median(py))
            # sort by nearest neighbours
            newnode.sort_neighbours()
            newnode.calc_local_density(n_neighbours)
            self.nodes[p] = newnode
        self.pids = list(self.pids)
        #print self.pids
        # sort nodes by local density
        self.pids = sorted(self.pids,key = lambda x: self.nodes[x].local_density, reverse=False  )
        # start with highest density node in the lattice: call that the primary node (0,0)
        #print self.pids
        self.nodes[self.pids[0]].x = 0.0
        self.nodes[self.pids[0]].y = 0.0
        for pid in self.pids:
            if self.nodes[pid].has_position(): # if node already has a position, use it to determine neighbour positions from averaged vectors
                self.assign_neighbour_position(pid,n_neighbours=n_neighbours)
        # now fill in any nodes without positions
        for pid in self.pids:
            if not self.nodes[pid].has_position():
                #print "%d has no position" % (pid)
                self.find_nearest_neighbour_with_position(pid)
        # now everyone should have a position in the formation
        # finally transform to com frame of lattice
        self.to_com_frame()
        self.calc_formation_bounds()
        self.calc_player_deviations_within_formation(match,nexclude=nexclude)
        # get average centre of mass of team throughout lattices
        self.calc_lattice_average_com()  
        # calculate start and end time of observations
        self.t_start = self.lattices[0].timestamp
        self.t_end = self.lattices[-1].timestamp
    
    def add_posession_data(self,posessions, period_start):
        offset = (period_start-1.0)*45.0
        self.team_pos_to_t_start = np.sum( [p.pos_duration for p in posessions if p.team==self.team and p.pos_start_matchtime<=(self.t_start+offset)] )
        self.team_pos_after_t_start = np.sum( [p.pos_duration for p in posessions if p.team==self.team and p.pos_start_matchtime>(self.t_start+offset)] )
    
    def calc_formation_bounds(self):
        x = []
        y = []
        for pid in self.pids:
            x.append( self.nodes[pid].x )
            y.append( self.nodes[pid].y )
        self.bounds = (min(x),max(x),min(y),max(y))
    
    def plot_formation(self,factor=1.0,offsets=(0,0),figax=None,labels=False,elipses=False,lt='ko',nsigma=1):
        if figax is None:
            fig,ax = plt.subplots()
        else:
            fig,ax = figax
        for pid in self.pids:
            x = factor*( self.nodes[pid].x )+offsets[0] 
            y = factor*( self.nodes[pid].y )+offsets[1] 
            mu = self.formation_distributions[pid][0]*factor
            cov = self.formation_distributions[pid][1]*factor*factor
            #ax.plot(x,y,lt,MarkerSize=10)
            ax.plot(mu[0],mu[1],lt,MarkerSize=10)
            if labels:
                ax.text(x+factor*0.5,y+factor*0.5,pid)
            if elipses:
                ax.plot(mu[0],mu[1],lt,MarkerSize=10)
                utils.plot_bivariate_normal(mu,cov,(fig,ax),nsigma=nsigma,lt=lt)

    
    def plot_cluster_formation(self,factor=1.0,offsets=(0,0),figax=None,points=False,elipses=False,lt='ko',nsigma=1):
        if figax is None:
            fig,ax = plt.subplots()
        else:
            fig,ax = figax
        for pid in self.pids:
            x = factor*( self.nodes[pid].x )+offsets[0] 
            y = factor*( self.nodes[pid].y )+offsets[1] 
            mu = self.cluster_formation_distributions[pid][0]*factor
            cov = self.cluster_formation_distributions[pid][1]*factor*factor
            if points:
                ax.plot(self.nodes_cluster[pid][:,0]*factor,self.nodes_cluster[pid][:,1]*factor,'.',alpha=0.5)
            if elipses:
                utils.plot_bivariate_normal(mu,cov,(fig,ax),nsigma=nsigma)
            ax.plot(x,y,lt,MarkerSize=10)
            ax.plot(mu[0],mu[1],'kd',MarkerSize=10)
            
    def get_xydata(self):
        xydata = []
        for p in self.pids:
            xydata.append([self.nodes[p].x,self.nodes[p].y])
        return np.array(xydata)
       

class lattice(object):
    # contains a lattice of player positions. Can be from a single frame, or an aggregated posession window
    def __init__(self,pos_type,team_parity,timestamp,players,exclude):
        self.pos_type = 1 if pos_type=='A' else -1
        self.period_sign = self.pos_type * team_parity
        self.players = players
        self.pids = [p for p in self.players.keys() if p not in exclude]
        self.timestamp = timestamp 
        self.exclude = exclude
        self.nodes = {}
        self.formation_offset_nodes = {}
        self.add_nodes()
        self.calc_com()
        self.calc_edges()
        self.to_com()
        
    def add_nodes(self):
        # player positions in formation
        for p in self.pids:
            px = self.players[p].pos_x/100.*self.period_sign
            py = self.players[p].pos_y/100.*self.period_sign
            self.nodes[p] = node(p,px,py)
            
    def add_formation_offset_nodes(self,pid,dx,dy):
        self.formation_offset_nodes[pid] = node(pid,dx,dy)
         
            
    def calc_edges(self):
        # calculate vectors between all the nodes
        for p in self.pids:
            for n in self.pids:
                if p!=n:
                    dx = self.nodes[n].x - self.nodes[p].x
                    dy = self.nodes[n].y - self.nodes[p].y
                    self.nodes[p].add_neighbour(self.nodes[n].pid,dx,dy)
    
    def calc_com(self):
        self.xcom = 0.
        self.ycom = 0.
        for p in self.pids:
            self.xcom += self.nodes[p].x
            self.ycom += self.nodes[p].y
        self.xcom = self.xcom/float(len(self.pids))
        self.ycom = self.ycom/float(len(self.pids))
        
    def get_com(self,exclude=[]):
        xcom = 0.
        ycom = 0.
        for p in self.pids:
            if p not in exclude:
                xcom += self.nodes[p].x + self.xcom
                ycom += self.nodes[p].y + self.ycom
        xcom = xcom/float(len(self.pids)-len(exclude))
        ycom = ycom/float(len(self.pids)-len(exclude))
        return xcom,ycom
     
    def from_com(self):
        for p in self.pids:
            self.nodes[p].x += self.xcom
            self.nodes[p].y += self.ycom
     
    def to_com(self):
        for p in self.pids:
            self.nodes[p].x -= self.xcom
            self.nodes[p].y -= self.ycom
        
    def get_xydata(self):
        xydata = []
        for p in self.pids:
            xydata.append([self.nodes[p].x,self.nodes[p].y])
        return np.array(xydata)
    

class node(object):
    # this is a single player position in a frame or formation
    def __init__(self,pid,px,py):
        self.pid = pid
        self.neighbours = {}
        self.x = px
        self.y = py
        
    def has_position(self):
        return ~np.isnan(self.x) and ~np.isnan(self.y)        
        
    def add_neighbour(self,pid,dx,dy):
        dist = np.sqrt(dx*dx+dy*dy)
        self.neighbours[pid] = (pid,dx,dy,dist)
        
    def sort_neighbours(self):
        self.nearest_neighours = [self.neighbours[p] for p in self.neighbours.keys()]
        self.nearest_neighours = sorted(self.nearest_neighours,key = lambda x: x[3])
     
    def calc_local_density(self,N):
        # average distance to N nearest neighbours
        #self.local_density = np.sum([x[3] for x in self.nearest_neighours[:N]])/float(N)
        self.local_density = self.nearest_neighours[N-1][3]
        #print self.local_density, self.nearest_neighours[:N]
     
    def __repr__(self):
        print self.pid
        