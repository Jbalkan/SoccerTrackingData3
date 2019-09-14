# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:19:49 2019

@author: laurieshaw
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:33:43 2019

@author: laurieshaw
"""

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import OPTA as opta
import OPTA_visuals as ovis


fpath = "/Users/laurieshaw/Documents/Football/Data/TrackingData/Tracab/SuperLiga/All/"
#fpath='/n/home03/lshaw/Tracking/Tracab/SuperLiga/' # path to directory of Tracab data

'''
Aalborg Match IDs
984455	SnderjyskE_vs_AalborgBK	
984460	AalborgBK_vs_FCMidtjylland	
984468	FCKbenhavn_vs_AalborgBK	
984476	Vendsyssel_vs_AalborgBK	
984481	AalborgBK_vs_FCNordsjlland	
984491	ACHorsens_vs_AalborgBK	
984495	AalborgBK_vs_EsbjergfB	
984505	RandersFC_vs_AalborgBK	
984509	AalborgBK_vs_VejleBK	
984518	HobroIK_vs_AalborgBK	
984523	AalborgBK_vs_OdenseBoldklub	
984530	AalborgBK_vs_BrndbyIF	
984539	AGFAarhus_vs_AalborgBK	
984544	AalborgBK_vs_Vendsyssel	
984554	FCNordsjlland_vs_AalborgBK	
984558	AalborgBK_vs_FCKbenhavn	
984570	VejleBK_vs_AalborgBK	
984575	EsbjergfB_vs_AalborgBK	
984579	AalborgBK_vs_ACHorsens	
984590	OdenseBoldklub_vs_AalborgBK	
984593	AalborgBK_vs_RandersFC	
984602	FCMidtjylland_vs_AalborgBK	
984607	AalborgBK_vs_SnderjyskE	
984614	AalborgBK_vs_HobroIK	
984623	BrndbyIF_vs_AalborgBK	
984628	AalborgBK_vs_AGFAarhus	
'''

team= 'Aalborg BK'
match_id = 984602
fname = str(match_id)

match_OPTA = opta.read_OPTA_f7(fpath,fname) 
match_OPTA = opta.read_OPTA_f24(fpath,fname,match_OPTA)

ovis.plot_all_shots(match_OPTA,plotly=False)
ovis.make_expG_timeline(match_OPTA)
#ovis.plot_defensive_actions(team,[match_OPTA],include_tackles=True,include_intercept=True)
