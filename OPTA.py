# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:21:38 2019

@author: laurieshaw
"""

import datetime as dt
from bs4 import BeautifulSoup
import data_utils as utils
import csv
import numpy as np
#import plotly.plotly as py
#import plotly.tools as tls
import matplotlib.pyplot as plt


def read_OPTA_f24(fpath,fname,match_opta):
    typeids,qualids = get_OPTA_id_descriptions()
    f24 = fpath + fname + '/' + 'f24.xml'
    # get meta data
    f = open(f24, "r")
    data = f.read()
    f.close()
    soup = BeautifulSoup(data, 'xml') # deal with xml markup (BS might be overkill here)
    game = soup.find('Game')
    match_opta.add_gamedata(game,match_opta,typeids,qualids)
    match_opta.hometeam.get_team_events(match_opta.events,match_opta.awayteam.team_id,match_opta)
    match_opta.awayteam.get_team_events(match_opta.events,match_opta.hometeam.team_id,match_opta)
    return match_opta
    
def read_OPTA_f7(fpath,fname):
    f7 = fpath + fname + '/' + 'f7.xml'
    # get meta data
    f = open(f7, "r")
    data = f.read()
    f.close()
    soup = BeautifulSoup(data, 'xml') # deal with xml markup (BS might be overkill here)
    competition = soup.find('Competition')
    matchdata = soup.find('MatchData')
    teamdata = soup.find_all('Team')
    venue = soup.find_all('Venue')
    match = OPTAmatch(matchdata)
    match.get_teams(teamdata)
    match.fpath = fpath + fname
    return match
    
def add_tracab_attributes(match_OPTA,match_tb):
    # pitch dimensions for plotting
    match_OPTA.fPitchXSizeMeters = match_tb.fPitchXSizeMeters
    match_OPTA.fPitchYSizeMeters = match_tb.fPitchYSizeMeters
    return match_OPTA

def get_OPTA_id_descriptions():
    # reads in descriptions of OPTAs and type and qualifier descriptors
    typeids_path = "/Users/laurieshaw/Documents/Football/Data/OPTA/TypeID_Descriptions.csv"
    typeids = {}
    with open(typeids_path, 'rU') as csvfile:
        refreader = csv.reader(csvfile, dialect = 'excel')
        refreader.next() # get rid of header
        for row in refreader: # first row is header
            typeids[int(row[0])] = row[1]
    qualids_path = "/Users/laurieshaw/Documents/Football/Data/OPTA/QualID_Descriptions.csv"
    qualids = {}
    with open(qualids_path, 'rU') as csvfile:
        refreader = csv.reader(csvfile, dialect = 'excel')
        refreader.next() # get rid of header
        for row in refreader: # first row is header
            qualids[int(row[0])] = row[1]
    return typeids,qualids

    
    
class OPTAmatch(object):
    # OPTA match class. holds all information about teams and events
    def __init__(self,matchdata):
        self.raw = matchdata
        self.set_match_date()
        # default pitch sizes. can be updated to match tracking data
        self.fPitchXSizeMeters = 105.0
        self.fPitchYSizeMeters = 68.0
        self.total_match_time = int( matchdata.find('Stat',{"Type" : "match_time"}).text )
        
    def set_match_date(self):
        self.datestring = self.raw.find('Date').text
        year = int(self.datestring[:4])
        month = int(self.datestring[4:6])
        day = int(self.datestring[6:8])
        self.date = dt.datetime(year=year,month=month,day=day)
    
    def get_teams(self,teamdata):
        assert len(teamdata)==2
        self.hometeam = OPTAteam(teamdata[0],'H')
        self.awayteam = OPTAteam(teamdata[1],'A')
        self.team_map = {}
        self.team_map[self.hometeam.team_id] = self.hometeam
        self.team_map[self.awayteam.team_id] = self.awayteam
        
    def add_gamedata(self,gamedata,match,typeids,qualids):
        self.homegoals = int(gamedata['home_score'])
        self.awaygoals = int(gamedata['away_score'])
        assert utils.remove_accents( gamedata['home_team_name'] ) == self.hometeam.teamname
        assert utils.remove_accents( gamedata['away_team_name'] ) == self.awayteam.teamname
        self.season = gamedata['season_id']
        date_check = dt.datetime.strptime( gamedata['game_date'][:10], '%Y-%m-%d')
        assert date_check==self.date
        eventdata = gamedata.find_all('Event')
        self.events = []
        for event in eventdata:
            self.events.append(OPTAevent(event,match,typeids,qualids))
        
    def __repr__(self):
        s = "%s vs %s on %s" % (self.hometeam.teamname, self.awayteam.teamname, self.datestring)
        return s
    
class OPTAteam(object):
    # team class to hold events for each team in a match
    def __init__(self,team,homeaway):
        self.raw = team
        self.homeaway = homeaway
        self.country = team.find('Country').text
        self.team_id = int( team['uID'][1:] )
        self.teamname = utils.remove_accents(  team.find('Name').text )
        self.players = []
        self.get_players()
        
    def get_players(self):
        playerdata =  self.raw.find_all('Player')
        for player in playerdata:
            self.players.append( OPTAplayer(player) )
        # now construct player map for easy identification of players
        self.player_map = {}
        for p in self.players:
            self.player_map[p.id] = p
            
    def get_team_events(self,events,opp_id,match_OPTA):
        self.events = []
        opp_events = []
        for e in events:
            if e.team_id == self.team_id:
                self.events.append(e)
            elif e.team_id == opp_id:
                opp_events.append(e)
        self.events = sorted( self.events, key = lambda x: x.time)
        for i,e in enumerate(self.events):
            if e.is_shot:
                e.get_assist(self.events,match_OPTA)
                e.get_preceeding_event(self.events[i-1])
                e.find_opponent_error(opp_events)
                e.calc_shot_xG_Caley()
                e.calc_shot_xG_Caley2()
            
    def __repr__(self):
        return self.teamname

class OPTAplayer(object):
    # player class
    def __init__(self,player):
        self.raw = player
        self.firstname = utils.remove_accents( player.find('PersonName').find('First').text )
        self.lastname = utils.remove_accents( player.find('PersonName').find('Last').text )
        self.position = player['Position']
        self.id = int( player['uID'][1:] )
        
    def __repr__(self):
        return "%s %s (%s)" % (self.firstname,self.lastname,self.position)
    
class OPTAevent(object): 
    # event class, including xG calculatons for several different models
    def __init__(self,event,match_OPTA,typeids,qualids):
        self.raw = event
        self.unique_id = int(event['id'])
        self.event_id = int(event['event_id'])
        self.type_id = int(event['type_id'])
        
        self.period_id = int(event['period_id'])
        self.min = int(event['min'])
        self.sec = int(event['sec'])
        self.time = float(self.min) + float(self.sec/60.)
        self.team_id = int(event['team_id'])
        self.team_name = match_OPTA.team_map[self.team_id].teamname
        self.home_away_team = match_OPTA.team_map[self.team_id].homeaway
        if 'player_id' in event.attrs.keys():
            self.player_id = int(event['player_id'])
            self.player_name = match_OPTA.team_map[self.team_id].player_map[self.player_id].lastname
        else:
            self.player_id = None
            self.player_name = 'N/A'
        self.x = (float(event['x'])/100.-0.5) # put the origin at the center of the pitch
        self.y = (float(event['y'])/100.-0.5) # put the origin at the center of the pitch
        self.outcome = int(event['outcome'])
        self.qualifiers = []
        self.qual_id_list = []
        qualdata = event.find_all('Q')
        for q in qualdata:
            self.qualifiers.append( OPTAqualifier(q,qualids) )
            self.qual_id_list.append(self.qualifiers[-1].qual_id)
        self.set_event_description(typeids)
        self.is_shot = self.type_id in [13,14,15,16]
        self.is_goal = self.type_id == 16
        if self.is_shot:
            self.expG_caley = -1 # until calculated
            self.calc_shot_xG_inputs(match_OPTA)
            self.calc_shot_xG()
            self.get_shot_descriptor(match_OPTA)
            
    def set_event_description(self,typeids):
        if self.type_id in typeids.keys():
            self.description = typeids[self.type_id]
        else:
            self.description = str(self.type_id )    
        if self.type_id == 15 and 82 in self.qual_id_list:
            self.description = 'Blocked'
        
    def get_assist(self,events,match_OPTA):
        self.assisted = False
        self.assist_throughball = False
        self.assist_cross = False
        self.second_assist_throughball = False
        self.second_assist_cross = False
        self.assist_cut_back = False
        self.assist_inv_distance_to_goal = 0.0
        self.assist_relative_angle_to_goal = 0.0
        if 55 in self.qual_id_list:
            self.assisted = True
            assist_event_id = int( self.qualifiers[self.qual_id_list.index(55)].value )
            related_events = [e for e in events if e.event_id==assist_event_id]
            if len(related_events)==0:
                print "no related event found"
            elif len(related_events)>1:
                print "multiple related events"
            else:
                self.assist = related_events[0]
            self.assist_throughball = 4 in self.assist.qual_id_list
            self.assist_cross = 2 in self.assist.qual_id_list
            self.assist_cut_back = 195 in self.assist.qual_id_list
            dist,angle = self.calc_caley2_dist_angle(self.assist, match_OPTA)
            self.assist_distance_to_goal = dist
            self.assist_relative_angle_to_goal = angle
            self.assist_inv_distance_to_goal = 1./self.assist_distance_to_goal
        if 216 in self.qual_id_list:
            self.second_assist = True
            assist_event_id = int( self.qualifiers[self.qual_id_list.index(216)].value )
            related_events = [e for e in events if e.event_id==assist_event_id]
            if len(related_events)==0:
                print "no related event found"
            elif len(related_events)>1:
                print "multiple related events"
            else:
                self.second_assist = related_events[0]
            self.second_assist_throughball = 4 in self.second_assist.qual_id_list
            self.second_assist_cross = 2 in self.second_assist.qual_id_list

    
    def get_preceeding_event(self,preceeding_event):
        self.preceeding_event = preceeding_event
        self.shot_off_rebound = 137 in self.preceeding_event.qual_id_list or 138 in self.preceeding_event.qual_id_list  or 101 in self.preceeding_event.qual_id_list   
        
    def find_opponent_error(self,opponent_events,t_window=6):
        # look for opponent defensive errors within t_window seconds of shot
        opp_events_window = [e for e in opponent_events if self.time-e.time<t_window/60.]
        self.opponent_error = [e for e in opp_events_window if 69 in e.qual_id_list]
        self.shot_following_error = len(self.opponent_error)>0
            
    def calc_shot_xG_inputs(self,match_OPTA):
        self.strat_to_m = 0.267*0.91
        # calculate distance from y=0 to posts in pitch units
        posts = 7.32 # 7.32 is width of goal line in m
        self.xgoal = (0.5-self.x)*match_OPTA.fPitchXSizeMeters# x-distance to opponents goal in m
        self.ygoal = self.y*match_OPTA.fPitchYSizeMeters# y-distance to center of opponents goal in m
        self.distance = np.sqrt(self.xgoal**2+self.ygoal**2) # distance to center of goal in m
        self.angle_to_centre = np.arccos(self.xgoal/self.distance)*180./np.pi
        # x distance to posts (nearest,furthest)
        x_to_posts = (np.abs(np.abs(self.ygoal)-posts),np.abs(self.ygoal)+posts)
        # some angles to posts
        alpha = np.arctan(x_to_posts[0]/self.xgoal)
        phi = np.arctan(x_to_posts[1]/self.xgoal)
        # opening angle to goal
        if np.abs(self.ygoal)>posts:
            self.angle_opening = (phi - alpha)*180/np.pi
        else:
            self.angle_opening = (phi + alpha)*180/np.pi 
        self.penalty = 9 in self.qual_id_list
        self.header = 15 in self.qual_id_list
        self.foot = 72 in self.qual_id_list or 20 in self.qual_id_list
        self.other_body_part = 21 in self.qual_id_list
        self.expG = -1 # not calculated yet
        self.individual_play = 215 in self.qual_id_list
        self.big_chance = 214 in self.qual_id_list
        self.direct_free_kick = 26 in self.qual_id_list
        self.fast_break = 22 in self.qual_id_list
        self.from_set_play = 24 in self.qual_id_list
        self.from_corner = 25 in self.qual_id_list
        self.direct_free_kick = 26 in self.qual_id_list

        if not (self.header or self.foot or self.other_body_part):
            print self
            assert False
            
    def calc_shot_xG(self,xG_pen = 0.76):
        params_f=[-0.3664,-0.0335/self.strat_to_m, 0.0160]
        params_h=[ 0.1722, -0.0796/self.strat_to_m, 0.0171]
        x = [1.,self.distance,self.angle_opening]
        if  self.penalty:
            self.expG = xG_pen
        elif self.foot:
            self.expG = 1./(1.+np.exp( -np.dot(params_f,x )))
        elif self.header or self.other_body_part:
            self.expG = 1./(1.+np.exp( -np.dot(params_h,x )))
    
    def calc_shot_xG_Caley(self,xG_pen = 0.75):
        # Michael Caley's first xG model: http://cartilagefreecaptain.sbnation.com/2014/9/11/6131661/premier-league-projections-2014#methoderology
        self.meters_to_yards = 1.094
        central_channel_width = 6.0/self.meters_to_yards/2. # central channel width in m
        if np.abs(self.ygoal) <= central_channel_width:
            relative_angle_to_goal = 1.0 
        else: 
            relative_angle_to_goal = np.arctan(self.xgoal/np.abs(self.ygoal))/np.arctan(self.xgoal/central_channel_width)
        adj_distance_to_goal = self.xgoal / relative_angle_to_goal**1.32
        if self.penalty:
            xG = xG_pen
            self.caley_types = 0
        elif self.header or self.other_body_part:
            if self.assist_cross:
                # Headers off crosses
                xG = 0.64 * np.exp(-0.21 * adj_distance_to_goal)
                self.caley_types = 1
            else:
                # Headers, not off crosses
                xG = 1.13 * np.exp(-0.29 * adj_distance_to_goal)
                self.caley_types = 2
        else:  # shots with feet 
            if self.assist_cross:
                # Non-headed shots off crosses
                xG = 0.96 * np.exp(-0.19 * adj_distance_to_goal)
                self.caley_types = 3
            elif self.shot_off_rebound:
                # Shots off a rebound
                xG = 0.94 * np.exp(-0.09 * adj_distance_to_goal)
                self.caley_types = 4
            elif self.individual_play: 
                # Shots following a successful dribble
                xG = 1.02 * np.exp(-0.12 * adj_distance_to_goal) + 0.21 * self.assist_throughball + 0.017
                self.caley_types = 5
            else:
                # Regular shots
                xG = 0.86 * np.exp(-0.14 * adj_distance_to_goal) + 0.15 * self.assist_throughball + 0.045 * self.direct_free_kick + 0.012    
                self.caley_types = 6
        self.expG_caley = xG 
        
    def calc_shot_xG_Caley2(self,xG_pen = 0.75):
        # Michael Caley's updated xG model: https://cartilagefreecaptain.sbnation.com/2015/10/19/9295905/premier-league-projections-and-new-expected-goals
        self.meters_to_yards = 1.094
        central_channel_width = 8.0/self.meters_to_yards/2. # central channel width in m
        if np.abs(self.ygoal) <= central_channel_width:
            relative_angle_to_goal = 1.0 
        else: 
            relative_angle_to_goal = np.arctan( self.xgoal/ ( np.abs(self.ygoal)-central_channel_width ) ) / (np.pi/2.)
        distance_to_goal = np.sqrt(self.xgoal**2+self.ygoal**2)# * self.meters_to_yards
        #distance_to_goal = self.xgoal * self.meters_to_yards
        inv_distance_to_goal = 1./distance_to_goal
        inv_rel_angle_to_goal = 1./relative_angle_to_goal
        inv_angle_distance_prod = inv_distance_to_goal*inv_rel_angle_to_goal
        
        if self.penalty:
            x = 1.1 # in logit gives xG = 0.75
            self.caley_types2 = 0
        elif self.direct_free_kick:
            x = -3.84 + -0.08*distance_to_goal + 98.7*inv_distance_to_goal + 3.54*inv_rel_angle_to_goal - 91.1*inv_angle_distance_prod
            self.caley_types2 = 1
        elif self.header or self.other_body_part:
            if self.assist_cross:
                # Headers off crosses
                x = -2.88 + -0.21*distance_to_goal + 2.13*relative_angle_to_goal + 4.31*self.assist_inv_distance_to_goal + 0.46*self.assist_relative_angle_to_goal
                x += 0.2*self.fast_break + -0.24*self.from_corner + 1.2*self.big_chance + 1.1*self.shot_following_error + -0.18*self.other_body_part
                self.caley_types2 = 2
            else:
                # Headers, not off crosses
                x = -3.85 + -0.1*distance_to_goal + 2.56*inv_distance_to_goal + 1.94*relative_angle_to_goal + 0.51*self.assist_throughball 
                x += 0.44*self.fast_break + 1.3*self.big_chance + 1.1*self.shot_following_error  + 0.7*self.shot_off_rebound + 1.14*self.other_body_part
                self.caley_types2 = 3
        else:  # shots with feet 
            if self.assist_cross:
                # Non-headed shots off crosses
                x = -2.8 + -0.11*distance_to_goal + 3.52*inv_distance_to_goal + 1.14*relative_angle_to_goal + 6.94*self.assist_inv_distance_to_goal + 0.59*self.assist_relative_angle_to_goal
                x += 0.24*self.fast_break + -0.12*self.from_corner + 1.25*self.big_chance + 1.1*self.shot_following_error 
                self.caley_types2 = 4
            else:
                # Regular shots
                x = -3.190 + -0.095*distance_to_goal + 3.180*inv_distance_to_goal + 1.880*relative_angle_to_goal + 0.240*inv_rel_angle_to_goal + -2.090*inv_angle_distance_prod
                x += 0.450*self.assist_throughball + 0.640*self.second_assist_throughball + 2.180*self.assist_inv_distance_to_goal + 0.120*self.assist_relative_angle_to_goal
                x += 0.23*self.fast_break + -0.18*self.from_corner + 1.2*self.big_chance + 1.1*self.shot_following_error + 0.390*self.individual_play + 0.370*self.shot_off_rebound
                self.caley_types2 = 5
        self.expG_caley2 = np.exp(x)/(1+np.exp(x))
    
    def calc_caley2_dist_angle(self,event, match_OPTA):
        meters_to_yards = 1.094        
        xgoal = (0.5-event.x)*match_OPTA.fPitchXSizeMeters# x-distance to opponents goal in m
        ygoal = event.y*match_OPTA.fPitchYSizeMeters# y-distance to center of opponents goal in m      
        #distance_to_goal = xgoal*meters_to_yards # is the Caley model in m or yards?
        distance_to_goal = np.sqrt(xgoal**2+ygoal**2)# * meters_to_yards
        central_channel_width = 8.0/meters_to_yards/2. # central channel width in m
        if np.abs(ygoal) <= central_channel_width:
            relative_angle_to_goal = 1.0 
        else: 
            relative_angle_to_goal = np.arctan( xgoal/ ( np.abs(ygoal)-central_channel_width ) ) / (np.pi/2.)
        return distance_to_goal,relative_angle_to_goal    
    
    def get_shot_descriptor(self,match_OPTA,max_char_line=40):
        self.shot_id = self.home_away_team + str(self.event_id)
        self.shot_descriptor = "%s (%s), %d min %d sec, xG: %1.2f\n" % (self.player_name,self.team_name,self.min,self.sec,self.expG_caley)
        sub_info = ""
        for q in self.qualifiers:
            if q.qual_id in [15,72,20,21,22,23,24,25,26,96,97]:
                sub_info += q.description
                if len(sub_info)>=max_char_line:
                    sub_info += '\n'
                else:
                    sub_info += ', '
        self.shot_descriptor += sub_info + self.description + '\nShot_id: ' + self.shot_id
        
            
    def __repr__(self):
        if self.is_shot:
            s = "%s, %1.2f, %s, xG: %1.2f" % (self.team_name,self.time,self.description,self.expG_caley)
        else:   
            s = "%s, %1.2f, %s" % (self.team_name,self.time,self.description)
        return s
        
class OPTAqualifier(object):
    # class holds qualifiers for each event
    def __init__(self,qualifier,qualids):
        self.unique_id = int(qualifier['id'])
        self.qual_id = int(qualifier['qualifier_id'])
        if self.qual_id in qualids.keys():
            self.description = qualids[self.qual_id]
        else:
            self.description = str(self.qual_id )
        if 'value' in qualifier.attrs.keys():
            try:
                self.value = float(qualifier['value'])
            except ValueError:
                self.value = qualifier['value']
        else:
            self.value = ""
        
                
    def __repr__(self):
        s = "%s,%s,%s" % (self.qual_id,self.description,str(self.value))
        return s
        
        
        