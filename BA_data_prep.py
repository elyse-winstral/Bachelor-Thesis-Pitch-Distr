import pandas as pd
from pybaseball import cache
from pybaseball import statcast
import numpy as np

cache.enable()
# pd.set_option('display.max_rows', None, 'display.max_columns', None)


# get batting order data frame for 2023 year
batting_order1 = pd.read_csv("1_batting_order_2023.csv")
batting_order2 = pd.read_csv("2_batting_order_2023.csv")

batting_order = pd.concat([batting_order1, batting_order2], ignore_index=True)
batting_order = batting_order.rename(columns = {'id':'batter'})

#batting_order.shape
# old batting order shape: 44 568 = 18 * 2476
# expect new:              43 740 = 18 * 2430 = 18 * 162 * 15

# Determine whether pitcher's team is Winning, Losing, or Drawing
def prep_wld (league_data):

    league_data['wld'] = league_data['fld_score']- league_data['bat_score']
    

    # loss = 1, draw = 2, win = 3
    league_data.loc[league_data.bat_score > league_data.fld_score, 'wld'] = 1
    league_data.loc[league_data.bat_score == league_data.fld_score, 'wld'] = 2
    league_data.loc[league_data.bat_score < league_data.fld_score, 'wld'] = 3
    league_data = league_data.drop(columns=['bat_score', 'fld_score'])

    return league_data

# Convert 'on_1b','on_2b', 'on_3b' to binary variables
def binary_bases (league_data):
    zeros = {'on_1b': 0, "on_2b": 0, "on_3b": 0}

    league_data = league_data.fillna(value=zeros)
    league_data.loc[league_data.on_1b > 0, 'on_1b'] = 1
    league_data.loc[league_data.on_2b > 0, 'on_2b'] = 1
    league_data.loc[league_data.on_3b > 0, 'on_3b'] = 1
    league_data[['on_1b', 'on_2b', 'on_3b']] = league_data[['on_1b', 'on_2b', 'on_3b']].astype(int)

    return league_data
    
# Convert handedness to binary variable: L = 0, R = 1
def binary_handed (league_data):

    # batter handedness
    league_data.loc[league_data.stand == 'L', 'stand'] = 0
    league_data.loc[league_data.stand == 'R', 'stand'] = 1
    league_data['stand'] = league_data['stand'].astype(int)

    # pitcher handedness
    league_data.loc[league_data.p_throws == 'L', 'p_throws'] = 0
    league_data.loc[league_data.p_throws == 'R', 'p_throws'] = 1
    league_data['p_throws'] = league_data['p_throws'].astype(int)

    return league_data


# Normalize strikezone i.e. strikezone = [0,1] x [=-0.5,0.5], note: the area of the strikezone is prone to change based on a batter's height
# Using the batting dependent data to get the top and bottom, and using the width of home plate as the width of the strikezone
# see: https://www.mlb.com/glossary/rules/strike-zone
def normed_strikezone (league_data):

    # resize strike zone and replace old coordinate system
    delta_x = 17/12 #the width is 17"

    delta_z = league_data['sz_top'] - league_data['sz_bot']


    new_x = league_data['plate_x'] / delta_x # + 0.5 # for easier overview ([0, 1] x [0, 1] as strikezone)
    new_z = (league_data['plate_z'].clip(0, None) - league_data['sz_bot']) / delta_z
    # clip(0, None) replaces all negative plate_z values with 0; in the normed version the negative z-coordinates will indicate the pitch was below the strikezone

    league_data['normed_x'] = new_x
    league_data['normed_z'] = new_z


    # use temp to confirm a few instances
    # temp = basic_info.loc[: , ['normed_x', 'normed_z', 'type', 'sz_top', 'sz_bot', 'plate_x', 'plate_z', 'balls', 'strikes']]
    # temp.head(n=20)

    league_data = league_data.drop(columns=['plate_x', 'plate_z', 'sz_top', 'sz_bot']) 
    # league_data = league_data.drop(columns=['plate_x']) #use this for confirming normed results in the z-axis

    return league_data

# 3 Previous pitch types and locations of a given pitcher
def prev_pitches (league_data):

    # make pitch types numerical - (only considering types that appear more than 1000 times)
    # numbering as follows: all fastballs, all offspeeds, all breaks (sliders and curveballs) than "others"
    league_data.loc[league_data.pitch_type == 'FF', 'pitch_type'] = 1
    league_data.loc[league_data.pitch_type == 'SI', 'pitch_type'] = 2
    league_data.loc[league_data.pitch_type == 'FC', 'pitch_type'] = 3

    league_data.loc[league_data.pitch_type == 'CH', 'pitch_type'] = 5
    league_data.loc[league_data.pitch_type == 'FS', 'pitch_type'] = 6

    league_data.loc[league_data.pitch_type == 'SL', 'pitch_type'] = 8
    league_data.loc[league_data.pitch_type == 'ST', 'pitch_type'] = 9
    league_data.loc[league_data.pitch_type == 'SV', 'pitch_type'] = 10

    league_data.loc[league_data.pitch_type == 'CU', 'pitch_type'] = 11
    league_data.loc[league_data.pitch_type == 'KC', 'pitch_type'] = 12

    league_data.loc[league_data.pitch_type == 'FA', 'pitch_type'] = 11
    league_data.loc[~league_data.pitch_type.isin(np.arange(12)), 'pitch_type'] = 12


    # league_data['pitch_type'] = league_data['pitch_type'].astype(int)

    # previous pitchers and games - this will be our comparison
    league_data['prev_game_pk'] = league_data['game_pk'].shift(-1)
    league_data['2_prev_game_pk'] = league_data['game_pk'].shift(-2)
    league_data['3_prev_game_pk'] = league_data['game_pk'].shift(-3)


    league_data['prev_pitcher'] = league_data['pitcher'].shift(-1)
    league_data['2_prev_pitcher'] = league_data['pitcher'].shift(-2)
    league_data['3_prev_pitcher'] = league_data['pitcher'].shift(-3)

    # previous pitch
    league_data['prev_pitch_type'] = league_data['pitch_type'].shift(-1)
    league_data['prev_x'] = league_data['normed_x'].shift(-1)
    league_data['prev_z'] = league_data['normed_z'].shift(-1)

    league_data['prev_pitch_type'] = (league_data['prev_game_pk'] == league_data['game_pk']) * (league_data['prev_pitcher'] == league_data['pitcher']) * league_data['prev_pitch_type']
    league_data['prev_x'] = (league_data['prev_game_pk'] == league_data['game_pk']) * (league_data['prev_pitcher'] == league_data['pitcher']) * league_data['prev_x']
    league_data['prev_z'] = (league_data['prev_game_pk'] == league_data['game_pk']) *(league_data['prev_pitcher'] == league_data['pitcher']) * league_data['prev_z']

    # 2nd previous pitch
    league_data['2_prev_pitch_type'] = league_data['pitch_type'].shift(-2)
    league_data['2_prev_x'] = league_data['normed_x'].shift(-2)
    league_data['2_prev_z'] = league_data['normed_z'].shift(-2)

    league_data['2_prev_pitch_type'] = (league_data['2_prev_game_pk'] == league_data['game_pk']) * (league_data['2_prev_pitcher'] == league_data['pitcher']) * league_data['2_prev_pitch_type']
    league_data['2_prev_x'] = (league_data['2_prev_pitcher'] == league_data['pitcher']) * league_data['2_prev_x']
    league_data['2_prev_z'] = (league_data['2_prev_pitcher'] == league_data['pitcher']) * league_data['2_prev_z']


    # 3rd previous pitch
    league_data['3_prev_pitch_type'] = league_data['pitch_type'].shift(-3)
    league_data['3_prev_x'] = league_data['normed_x'].shift(-3)
    league_data['3_prev_z'] = league_data['normed_z'].shift(-3)

    league_data['3_prev_pitch_type'] = (league_data['3_prev_game_pk'] == league_data['game_pk']) * (league_data['3_prev_pitcher'] == league_data['pitcher']) * league_data['3_prev_pitch_type']
    league_data['3_prev_x'] = (league_data['3_prev_pitcher'] == league_data['pitcher']) * league_data['3_prev_x']
    league_data['3_prev_z'] = (league_data['3_prev_pitcher'] == league_data['pitcher']) * league_data['3_prev_z']


    league_data = league_data.fillna(-5) #if we set this to 0 we won't be able to distinguish the outliers as easy

    league_data = league_data.drop(columns= [ 'pitcher', 'pitch_type', 'prev_game_pk', '2_prev_game_pk', '3_prev_game_pk', 'prev_pitcher', '2_prev_pitcher', '3_prev_pitcher' ])

    return(league_data)


# 2023 season: 03-30-2023 - 10-01-2023 (Mar. 30th to Oct. 1st), prepare features and targets
def prep_data (start_dt, end_dt):

  # get lump data, extract all columns necessary for constructing the features
    league_data = statcast(start_dt, end_dt, parallel = True)
    league_data = league_data.loc[:, ['game_pk','sz_top', 'sz_bot', 'plate_x', 'plate_z', 'inning', 'bat_score', 'fld_score', 
                                      'on_1b', 'on_2b', 'on_3b', 'balls', 'strikes', 'outs_when_up', 'stand', 'pitcher', 'p_throws', 'batter', 'pitch_type']]

    league_data = prep_wld(league_data)
    league_data = binary_bases(league_data)
    league_data = binary_handed(league_data)
    league_data = normed_strikezone(league_data)
    league_data = prev_pitches(league_data)
    
    # league_data = adjusted_handedness(league_data)

    # merge data from statcast with batting order
    league_data = pd.merge(league_data, batting_order, on = ['game_pk', 'batter'], how = 'left' ) 
    league_data['batting_order'].fillna(0, inplace = True)
    league_data['batting_order'] = league_data['batting_order'].astype(int)
    league_data = league_data.drop(columns= ['fullName', 'batter', 'game_pk'])


    return(league_data)

league_data = prep_data('2023-03-29', '2023-10-01')
league_data.to_csv("Expanded_Features.csv")