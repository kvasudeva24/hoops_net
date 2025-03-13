import numpy as np
import pandas as pd
import tensorflow as tf

from nba_api.stats.endpoints import leaguedashteamstats as team
from nba_api.stats.endpoints import leaguedashteamshotlocations as shootingAdv
from nba_api.stats.endpoints import leaguedashteamptshot as shooting

'''
What we wanna use as our models weights


TEAMID  --> Advanced (TEAM_ID)
TEAMNAME  --> Advanced (TEAM_NAME)


Effective shooting percentage -->Advanced (EFG_PCT)
net rating -->Advanced (NET_RATING)
margin of victory  --> teamDashBoard (PLUS_MINUS)
offensive rating --> Advanced (OFF_RATING)
percetnage of rim --> ShotLocations(Restricted Area_FG_PCT)
percenatge of 16ft to 3 --> ShotLocations(Mid-Range_FG_PCT)
3pt percentage --> 
'''


years = ["2009-10","2010-11", "2011-12", "2012-13"]
for year in years:
    #deals with the advanced stats listed above
    #need to obtain the differnet columns that I need
    advanced_stats = team.LeagueDashTeamStats(season=year, measure_type_detailed_defense="Advanced")
    df_advanced = advanced_stats.get_data_frames()[0]



    #deals with the ShotLocation stats obtained above. Can extract as numpy array
    adv_shot_data = shootingAdv.LeagueDashTeamShotLocations(
        season=year,
        season_type_all_star="Regular Season"
    )
    df_adv_shooting = adv_shot_data.get_data_frames()[0]
    df_adv_shooting.columns = ['_'.join(col).strip() for col in df_adv_shooting.columns]
    df_adv_shooting = df_adv_shooting[[ "Restricted Area_FG_PCT", "Mid-Range_FG_PCT"]]



    #deals with 3pt shooting mentioned above
    shot_data = shooting.LeagueDashTeamPtShot(
        season=year,
        season_type_all_star="Regular Season"
    )

    df_shot_data = shot_data.get_data_frames()[0]


    #todo: grab all necessary columns into dataframe and export as csv
    team_id = df_advanced['TEAM_ID']
    team_name = df_advanced['TEAM_NAME']
    efg_pct = df_advanced['EFG_PCT']
    net_rating = df_advanced['NET_RATING']
    ts_pct_rank = df_advanced['TS_PCT_RANK']
    off_rating = df_advanced['OFF_RATING']
    percentage_of_rim = df_adv_shooting['Restricted Area_FG_PCT']
    percentage_of_16ft_to_3 = df_adv_shooting['Mid-Range_FG_PCT']
    three_pt_percentage = df_shot_data['FG3_PCT']

    final_df = pd.DataFrame({
        'TEAM_ID': team_id,
        'TEAM_NAME': team_name,
        'EFG_PCT': efg_pct,
        'NET_RATING': net_rating,
        'TS_PCT_RANK': ts_pct_rank,
        'OFF_RATING': off_rating,
        'Restricted Area_FG_PCT': percentage_of_rim,
        'Mid-Range_FG_PCT': percentage_of_16ft_to_3,
        'FG3_PCT': three_pt_percentage
    })

    final_df.to_csv(f"{year}_team_stats.csv", index=False)


#finished gathering data for the last 10 years based on our statistics into csvs will move onto making the model next week
#will store all csvs in a folder called team_stats