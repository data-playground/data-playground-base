#!/usr/bin/env python
# coding: utf-8

# In[1]:

import itertools
import json
import operator
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import lxml.html as LH
import pandas as pd
import requests
from google.cloud import bigquery

# %%

@dataclass
class NBA:

    def __init__(self, game_date = None, game_set = {}):
        # Dictionary to provide default names for expected outputs as well as the description, a default BigQuery table name and schema structure for processes
        self.VAR_NAMING_GUIDE = {
            'PLAYERS': {
                'bigquery_table': 'players',
                'description': 'High-level information on all players, such as name, status (playing or retired), current team...',
                'endpoint': 'https://stats.nba.com/stats/commonallplayers?',
                'version': 2,
                'parameters': {
                    'LeagueID': '00'
                },
                'table_name': 'CommonAllPlayers',
                'fields': {'PERSON_ID': "INTEGER",'DISPLAY_LAST_COMMA_FIRST': "STRING",'DISPLAY_FIRST_LAST': "STRING",'ROSTERSTATUS': "INTEGER",'FROM_YEAR': "STRING",'TO_YEAR': "STRING",'PLAYERCODE': "STRING",'PLAYER_SLUG': "STRING",'TEAM_ID': "INTEGER",'GAMES_PLAYED_FLAG': "STRING",'OTHERLEAGUE_EXPERIENCE_CH': "STRING"},
            },
            'PLAYER_DETAIL': {
                'bigquery_table': 'player_detail',
                'description': 'More detailed information on a specific player, such as weight, height, birthdate, college, seasons played (including G-League/D-League)',
                'endpoint': 'https://stats.nba.com/stats/commonplayerinfo?',
                'version': 2,
                'parameters': {
                    'LeagueID': '00', 
                    'PlayerID': '1628983'
                },
                'table_name': 'CommonPlayerInfo',
                'fields': {'PERSON_ID': "INTEGER",'FIRST_NAME': "STRING",'LAST_NAME': "STRING",'PLAYER_SLUG': "STRING",'BIRTHDATE': "STRING",'SCHOOL': "STRING",'COUNTRY': "STRING",'LAST_AFFILIATION': "STRING",'HEIGHT': "STRING",'WEIGHT': "STRING",'SEASON_EXP': "INTEGER",'JERSEY': "STRING",'POSITION': "STRING",'ROSTERSTATUS': "STRING",'DLEAGUE_FLAG': "STRING",'NBA_FLAG': "STRING",'GAMES_PLAYED_FLAG': "STRING",'DRAFT_YEAR': "STRING",'DRAFT_ROUND': "STRING",'DRAFT_NUMBER': "STRING",'GREATEST_75_FLAG': "STRING"},
            },
            'GAMES': {
                'bigquery_table': 'game_details',
                'description': 'General data on the game, including season, team and game identification, as well as game date',
                'endpoint': 'https://stats.nba.com/stats/leaguegamefinder?',
                'version': 2,
                'parameters': {
                    'Season': '2023-24', 
                    'LeagueID': '00'
                },
                'table_name': 'LeagueGameFinderResults',
                'fields': {'SEASON_ID': "STRING",'TEAM_ID': "INTEGER",'TEAM_ABBREVIATION': "STRING",'TEAM_NAME': "STRING",'GAME_ID': "STRING",'GAME_DATE': "STRING", "WL": "STRING"},
            },
            'GAME_DATA': {
                'bigquery_table': 'game_data',
                'description': 'High-level detail for a game, including broadcaster and arena. Great to be used when trying to get gameId for a specific date',
                'endpoint': 'https://stats.nba.com/stats/scoreboardv2?',
                'version': 2,
                'parameters': {
                    'GameDate': '2024-06-17', 
                    'LeagueID': '00'
                },
                'table_name': 'GameHeader',
                'fields': {'GAME_DATE_EST': "STRING",'GAME_SEQUENCE': "INTEGER",'GAME_ID': "STRING",'GAME_STATUS_ID': "INTEGER",'GAME_STATUS_TEXT': "STRING",'NATL_TV_BROADCASTER_ABBREVIATION': "STRING",'HOME_TV_BROADCASTER_ABBREVIATION': "STRING",'AWAY_TV_BROADCASTER_ABBREVIATION': "STRING",'ARENA_NAME': "STRING"},
            },
            'BS_SUMMARY': {
                'bigquery_table': 'boxscore_summary',
                'description': 'Detailed information on the game, inclduing: game time, duration, attendance, arena information, officials, brodcasters and high-level player data',
                'endpoint': 'https://stats.nba.com/stats/boxscoresummaryv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreSummary',
                'fields': {'gameId': "INTEGER", 'gameStatus': "INTEGER", 'gameStatusText': "STRING", 'period': "INTEGER", 'gameEt': "TIMESTAMP", 'duration': "STRING", 'attendance': "INTEGER", 'sellout': "INTEGER", 'gameLabel': "STRING", 'gameSubLabel': "STRING", 'seriesText': "STRING", 'isNeutral': "BOOLEAN", 'arena': {'arenaId': "INTEGER", 'arenaName': "STRING", 'arenaCity': "STRING", 'arenaState': "STRING", 'arenaCountry': "STRING", 'arenaTimezone': "STRING"}, 'officials': [{'personId': "INTEGER", 'firstName': "STRING", 'familyName': "STRING", 'jerseyNum': "INTEGER"}], 'broadcasters': {'nationalBroadcasters': [{'broadcasterId': "INTEGER", 'broadcastDisplay': "STRING", 'broadcasterDisplay': "STRING", 'broadcasterTeamId': "INTEGER"}], 'homeTvBroadcasters': [{'broadcasterId': "INTEGER", 'broadcastDisplay': "STRING", 'broadcasterDisplay': "STRING", 'broadcasterTeamId': "INTEGER"}], 'awayTvBroadcasters': [{'broadcasterId': "INTEGER", 'broadcastDisplay': "STRING", 'broadcasterDisplay': "STRING", 'broadcasterTeamId': "INTEGER"}]}, 'homeTeam': {'teamId': "INTEGER", 'teamWins': "INTEGER", 'teamLosses': "INTEGER", 'score': "INTEGER", 'seed': "INTEGER", 'players': [{'personId': "INTEGER", 'jerseyNum': "INTEGER"}]}, 'awayTeam': {'teamId': "INTEGER", 'teamWins': "INTEGER", 'teamLosses': "INTEGER", 'score': "INTEGER", 'seed': "INTEGER", 'players': [{'personId': "INTEGER", 'jerseyNum': "INTEGER"}]}, 'hustleStatus': "INTEGER"},
            },
            'BS_TRAD': {
                'bigquery_table': 'boxscore_trad',
                'description': 'Traditional option on the box-score. Included data like: points, rebounds, assists, fouls...',
                'endpoint': 'https://stats.nba.com/stats/boxscoretraditionalv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreTraditional',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'fieldGoalsMade': "INTEGER", 'fieldGoalsAttempted': "INTEGER", 'fieldGoalsPercentage': "FLOAT", 'threePointersMade': "INTEGER", 'threePointersAttempted': "INTEGER", 'threePointersPercentage': "FLOAT", 'freeThrowsMade': "INTEGER", 'freeThrowsAttempted': "INTEGER", 'freeThrowsPercentage': "FLOAT", 'reboundsOffensive': "INTEGER", 'reboundsDefensive': "INTEGER", 'reboundsTotal': "INTEGER", 'assists': "INTEGER", 'steals': "INTEGER", 'blocks': "INTEGER", 'turnovers': "INTEGER", 'foulsPersonal': "INTEGER", 'points': "INTEGER", 'plusMinusPoints': "FLOAT"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'fieldGoalsMade': "INTEGER", 'fieldGoalsAttempted': "INTEGER", 'fieldGoalsPercentage': "FLOAT", 'threePointersMade': "INTEGER", 'threePointersAttempted': "INTEGER", 'threePointersPercentage': "FLOAT", 'freeThrowsMade': "INTEGER", 'freeThrowsAttempted': "INTEGER", 'freeThrowsPercentage': "FLOAT", 'reboundsOffensive': "INTEGER", 'reboundsDefensive': "INTEGER", 'reboundsTotal': "INTEGER", 'assists': "INTEGER", 'steals': "INTEGER", 'blocks': "INTEGER", 'turnovers': "INTEGER", 'foulsPersonal': "INTEGER", 'points': "INTEGER", 'plusMinusPoints': "FLOAT"}}]}},
            },
            'BS_ADV': {
                'bigquery_table': 'boxscore_adv',
                'description': 'Advanced option on the box-score. Included data like: offensive rating, defensive rating, net rating, assist to turnover ratio...',
                'endpoint': 'https://stats.nba.com/stats/boxscoreadvancedv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreAdvanced',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'estimatedOffensiveRating': "FLOAT", 'offensiveRating': "FLOAT", 'estimatedDefensiveRating': "FLOAT", 'defensiveRating': "FLOAT", 'estimatedNetRating': "FLOAT", 'netRating': "FLOAT", 'assistPercentage': "FLOAT", 'assistToTurnover': "FLOAT", 'assistRatio': "FLOAT", 'offensiveReboundPercentage': "FLOAT", 'defensiveReboundPercentage': "FLOAT", 'reboundPercentage': "FLOAT", 'turnoverRatio': "FLOAT", 'effectiveFieldGoalPercentage': "FLOAT", 'trueShootingPercentage': "FLOAT", 'usagePercentage': "FLOAT", 'estimatedUsagePercentage': "FLOAT", 'estimatedPace': "FLOAT", 'pace': "FLOAT", 'pacePer40': "FLOAT", 'possessions': "FLOAT", 'PIE': "FLOAT"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'estimatedOffensiveRating': "FLOAT", 'offensiveRating': "FLOAT", 'estimatedDefensiveRating': "FLOAT", 'defensiveRating': "FLOAT", 'estimatedNetRating': "FLOAT", 'netRating': "FLOAT", 'assistPercentage': "FLOAT", 'assistToTurnover': "FLOAT", 'assistRatio': "FLOAT", 'offensiveReboundPercentage': "FLOAT", 'defensiveReboundPercentage': "FLOAT", 'reboundPercentage': "FLOAT", 'turnoverRatio': "FLOAT", 'effectiveFieldGoalPercentage': "FLOAT", 'trueShootingPercentage': "FLOAT", 'usagePercentage': "FLOAT", 'estimatedUsagePercentage': "FLOAT", 'estimatedPace': "FLOAT", 'pace': "FLOAT", 'pacePer40': "FLOAT", 'possessions': "FLOAT", 'PIE': "FLOAT"}}]}},
            },
            'BS_MISC': {
                'bigquery_table': 'boxscore_misc',
                'description': 'Miscellaneous option on the box-score. Included data like: contested shots, charges drawned, box outs...',
                'endpoint': 'https://stats.nba.com/stats/boxScoreMiscv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreMisc',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'pointsOffTurnovers': "INTEGER", 'pointsSecondChance': "INTEGER", 'pointsFastBreak': "INTEGER", 'pointsPaint': "INTEGER", 'oppPointsOffTurnovers': "INTEGER", 'oppPointsSecondChance': "INTEGER", 'oppPointsFastBreak': "INTEGER", 'oppPointsPaint': "INTEGER", 'blocks': "INTEGER", 'blocksAgainst': "INTEGER", 'foulsPersonal': "INTEGER", 'foulsDrawn': "INTEGER"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'pointsOffTurnovers': "INTEGER", 'pointsSecondChance': "INTEGER", 'pointsFastBreak': "INTEGER", 'pointsPaint': "INTEGER", 'oppPointsOffTurnovers': "INTEGER", 'oppPointsSecondChance': "INTEGER", 'oppPointsFastBreak': "INTEGER", 'oppPointsPaint': "INTEGER", 'blocks': "INTEGER", 'blocksAgainst': "INTEGER", 'foulsPersonal': "INTEGER", 'foulsDrawn': "INTEGER"}}]}},
            },
            'BS_SCORE': {
                'bigquery_table': 'boxscore_scoring',
                'description': 'Scoring option on the box-score. Included data like: % FG2A, % FG3A, % points on fast break, % assisted 3PT, % unassisted 3PT...',
                'endpoint': 'https://stats.nba.com/stats/boxscorescoringv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreScoring',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'percentageFieldGoalsAttempted2pt': "FLOAT", 'percentageFieldGoalsAttempted3pt': "FLOAT", 'percentagePoints2pt': "FLOAT", 'percentagePointsMidrange2pt': "FLOAT", 'percentagePoints3pt': "FLOAT", 'percentagePointsFastBreak': "FLOAT", 'percentagePointsFreeThrow': "FLOAT", 'percentagePointsOffTurnovers': "FLOAT", 'percentagePointsPaint': "FLOAT", 'percentageAssisted2pt': "FLOAT", 'percentageUnassisted2pt': "FLOAT", 'percentageAssisted3pt': "FLOAT", 'percentageUnassisted3pt': "FLOAT", 'percentageAssistedFGM': "FLOAT", 'percentageUnassistedFGM': "FLOAT"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'percentageFieldGoalsAttempted2pt': "FLOAT", 'percentageFieldGoalsAttempted3pt': "FLOAT", 'percentagePoints2pt': "FLOAT", 'percentagePointsMidrange2pt': "FLOAT", 'percentagePoints3pt': "FLOAT", 'percentagePointsFastBreak': "FLOAT", 'percentagePointsFreeThrow': "FLOAT", 'percentagePointsOffTurnovers': "FLOAT", 'percentagePointsPaint': "FLOAT", 'percentageAssisted2pt': "FLOAT", 'percentageUnassisted2pt': "FLOAT", 'percentageAssisted3pt': "FLOAT", 'percentageUnassisted3pt': "FLOAT", 'percentageAssistedFGM': "FLOAT", 'percentageUnassistedFGM': "FLOAT"}}]}},
            },
            'BS_USAGE': {
                'bigquery_table': 'boxscore_usage',
                'description': 'Usage option on the box-score. Included data like: usage percentage, % FGM, % FGA, % fouls drawned...',
                'endpoint': 'https://stats.nba.com/stats/boxscoreusagev3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreUsage',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'usagePercentage': "FLOAT", 'percentageFieldGoalsMade': "FLOAT", 'percentageFieldGoalsAttempted': "FLOAT", 'percentageThreePointersMade': "FLOAT", 'percentageThreePointersAttempted': "FLOAT", 'percentageFreeThrowsMade': "FLOAT", 'percentageFreeThrowsAttempted': "FLOAT", 'percentageReboundsOffensive': "FLOAT", 'percentageReboundsDefensive': "FLOAT", 'percentageReboundsTotal': "FLOAT", 'percentageAssists': "FLOAT", 'percentageTurnovers': "FLOAT", 'percentageSteals': "FLOAT", 'percentageBlocks': "FLOAT", 'percentageBlocksAllowed': "FLOAT", 'percentagePersonalFouls': "FLOAT", 'percentagePersonalFoulsDrawn': "FLOAT", 'percentagePoints': "FLOAT"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'usagePercentage': "FLOAT", 'percentageFieldGoalsMade': "FLOAT", 'percentageFieldGoalsAttempted': "FLOAT", 'percentageThreePointersMade': "FLOAT", 'percentageThreePointersAttempted': "FLOAT", 'percentageFreeThrowsMade': "FLOAT", 'percentageFreeThrowsAttempted': "FLOAT", 'percentageReboundsOffensive': "FLOAT", 'percentageReboundsDefensive': "FLOAT", 'percentageReboundsTotal': "FLOAT", 'percentageAssists': "FLOAT", 'percentageTurnovers': "FLOAT", 'percentageSteals': "FLOAT", 'percentageBlocks': "FLOAT", 'percentageBlocksAllowed': "FLOAT", 'percentagePersonalFouls': "FLOAT", 'percentagePersonalFoulsDrawn': "FLOAT", 'percentagePoints': "FLOAT"}}]}},
            },
            'BS_FOUR': {
                'bigquery_table': 'boxscore_fourfac',
                'description': 'Four Factors option on the box-score. Included data like: effective FGP, FTA, offensive rebound percentage...',
                'endpoint': 'https://stats.nba.com/stats/boxscorefourfactorsv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreFourFactors',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'effectiveFieldGoalPercentage': "FLOAT", 'freeThrowAttemptRate': "FLOAT", 'teamTurnoverPercentage': "FLOAT", 'offensiveReboundPercentage': "FLOAT", 'oppEffectiveFieldGoalPercentage': "FLOAT", 'oppFreeThrowAttemptRate': "FLOAT", 'oppTeamTurnoverPercentage': "FLOAT", 'oppOffensiveReboundPercentage': "FLOAT"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'effectiveFieldGoalPercentage': "FLOAT", 'freeThrowAttemptRate': "FLOAT", 'teamTurnoverPercentage': "FLOAT", 'offensiveReboundPercentage': "FLOAT", 'oppEffectiveFieldGoalPercentage': "FLOAT", 'oppFreeThrowAttemptRate': "FLOAT", 'oppTeamTurnoverPercentage': "FLOAT", 'oppOffensiveReboundPercentage': "FLOAT"}}]}},
            },
            'BS_TRACK': {
                'bigquery_table': 'boxscore_tracking',
                'description': 'Tracking option on the box-score. Included data like: distancem touches, secondary and free throw assists...',
                'endpoint': 'https://stats.nba.com/stats/boxscoreplayertrackv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScorePlayerTrack',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'distance': "FLOAT", 'reboundChancesOffensive': "INTEGER", 'reboundChancesDefensive': "INTEGER", 'reboundChancesTotal': "INTEGER", 'touches': "INTEGER", 'secondaryAssists': "INTEGER", 'freeThrowAssists': "INTEGER", 'passes': "INTEGER", 'assists': "INTEGER", 'contestedFieldGoalsMade': "INTEGER", 'contestedFieldGoalsAttempted': "INTEGER", 'contestedFieldGoalPercentage': "FLOAT", 'uncontestedFieldGoalsMade': "INTEGER", 'uncontestedFieldGoalsAttempted': "INTEGER", 'uncontestedFieldGoalsPercentage': "FLOAT", 'fieldGoalPercentage': "FLOAT", 'defendedAtRimFieldGoalsMade': "INTEGER", 'defendedAtRimFieldGoalsAttempted': "INTEGER", 'defendedAtRimFieldGoalPercentage': "FLOAT"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'distance': "FLOAT", 'reboundChancesOffensive': "INTEGER", 'reboundChancesDefensive': "INTEGER", 'reboundChancesTotal': "INTEGER", 'touches': "INTEGER", 'secondaryAssists': "INTEGER", 'freeThrowAssists': "INTEGER", 'passes': "INTEGER", 'assists': "INTEGER", 'contestedFieldGoalsMade': "INTEGER", 'contestedFieldGoalsAttempted': "INTEGER", 'contestedFieldGoalPercentage': "FLOAT", 'uncontestedFieldGoalsMade': "INTEGER", 'uncontestedFieldGoalsAttempted': "INTEGER", 'uncontestedFieldGoalsPercentage': "FLOAT", 'fieldGoalPercentage': "FLOAT", 'defendedAtRimFieldGoalsMade': "INTEGER", 'defendedAtRimFieldGoalsAttempted': "INTEGER", 'defendedAtRimFieldGoalPercentage': "FLOAT"}}]}},
            },
            'BS_HUSTLE': {
                'bigquery_table': 'boxscore_hustle',
                'description': 'Hustle option on the box-score. Included data like: contested shots, charges drawned, box outs...',
                'endpoint': 'https://stats.nba.com/stats/boxscorehustlev2?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreHustle',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'points': "INTEGER", 'contestedShots': "INTEGER", 'contestedShots2pt': "INTEGER", 'contestedShots3pt': "INTEGER", 'deflections': "INTEGER", 'chargesDrawn': "INTEGER", 'screenAssists': "INTEGER", 'screenAssistPoints': "INTEGER", 'looseBallsRecoveredOffensive': "INTEGER", 'looseBallsRecoveredDefensive': "INTEGER", 'looseBallsRecoveredTotal': "INTEGER", 'offensiveBoxOuts': "INTEGER", 'defensiveBoxOuts': "INTEGER", 'boxOutPlayerTeamRebounds': "INTEGER", 'boxOutPlayerRebounds': "INTEGER", 'boxOuts': "INTEGER"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'minutes': "STRING", 'points': "INTEGER", 'contestedShots': "INTEGER", 'contestedShots2pt': "INTEGER", 'contestedShots3pt': "INTEGER", 'deflections': "INTEGER", 'chargesDrawn': "INTEGER", 'screenAssists': "INTEGER", 'screenAssistPoints': "INTEGER", 'looseBallsRecoveredOffensive': "INTEGER", 'looseBallsRecoveredDefensive': "INTEGER", 'looseBallsRecoveredTotal': "INTEGER", 'offensiveBoxOuts': "INTEGER", 'defensiveBoxOuts': "INTEGER", 'boxOutPlayerTeamRebounds': "INTEGER", 'boxOutPlayerRebounds': "INTEGER", 'boxOuts': "INTEGER"}}]}}
            },
            'BS_MATCH': {
                'bigquery_table': 'boxscore_matchup',
                'description': 'Matchup option on the box-score. Not every game has this data available. Included data like: % of total offensive time, minutes matched up, help blocks...',
                'endpoint': 'https://stats.nba.com/stats/boxscorematchupsv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreMatchups',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'matchups': [{'personId': "INTEGER", 'statistics': {'matchupMinutes': "STRING", 'matchupMinutesSort': "FLOAT", 'PartialPossessions': "FLOAT", 'percentageDefenderTotalTime': "FLOAT", 'percentageOffensiveTotalTime': "FLOAT", 'percentageTotalTimeBothOn': "FLOAT", 'switchesOn': "INTEGER", 'playerPoints': "INTEGER", 'teamPoints': "INTEGER", 'matchupAssists': "INTEGER", 'matchupPotentialAssists': "INTEGER", 'matchupTurnovers': "INTEGER", 'matchupBlocks': "INTEGER", 'matchupFieldGoalsMade': "INTEGER", 'matchupFieldGoalsAttempted': "INTEGER", 'matchupFieldGoalsPercentage': "FLOAT", 'matchupThreePointersMade': "INTEGER", 'matchupThreePointersAttempted': "INTEGER", 'matchupThreePointersPercentage': "FLOAT", 'helpBlocks': "INTEGER", 'helpFieldGoalsMade': "INTEGER", 'helpFieldGoalsAttempted': "INTEGER", 'helpFieldGoalsPercentage': "FLOAT", 'matchupFreeThrowsMade': "INTEGER", 'matchupFreeThrowsAttempted': "INTEGER", 'shootingFouls': "INTEGER"}}]}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'matchups': [{'personId': "INTEGER", 'statistics': {'matchupMinutes': "STRING", 'matchupMinutesSort': "FLOAT", 'PartialPossessions': "FLOAT", 'percentageDefenderTotalTime': "FLOAT", 'percentageOffensiveTotalTime': "FLOAT", 'percentageTotalTimeBothOn': "FLOAT", 'switchesOn': "INTEGER", 'playerPoints': "INTEGER", 'teamPoints': "INTEGER", 'matchupAssists': "INTEGER", 'matchupPotentialAssists': "INTEGER", 'matchupTurnovers': "INTEGER", 'matchupBlocks': "INTEGER", 'matchupFieldGoalsMade': "INTEGER", 'matchupFieldGoalsAttempted': "INTEGER", 'matchupFieldGoalsPercentage': "FLOAT", 'matchupThreePointersMade': "INTEGER", 'matchupThreePointersAttempted': "INTEGER", 'matchupThreePointersPercentage': "FLOAT", 'helpBlocks': "INTEGER", 'helpFieldGoalsMade': "INTEGER", 'helpFieldGoalsAttempted': "INTEGER", 'helpFieldGoalsPercentage': "FLOAT", 'matchupFreeThrowsMade': "INTEGER", 'matchupFreeThrowsAttempted': "INTEGER", 'shootingFouls': "INTEGER"}}]}]}}
            },
            'BS_DEF': {
                'bigquery_table': 'boxscore_defense',
                'description': 'Defense option on the box-score. Not every game has this data available. Included data like: partial possessions, matchup assists, matchup turnovers...',
                'endpoint': 'https://stats.nba.com/stats/boxscoredefensivev2?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105'
                },
                'table_name': 'boxScoreDefensive',
                'fields': {'gameId': "INTEGER", 'homeTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'matchupMinutes': "STRING", 'partialPossessions': "FLOAT", 'switchesOn': "INTEGER", 'playerPoints': "INTEGER", 'defensiveRebounds': "INTEGER", 'matchupAssists': "INTEGER", 'matchupTurnovers': "INTEGER", 'steals': "INTEGER", 'blocks': "INTEGER", 'matchupFieldGoalsMade': "INTEGER", 'matchupFieldGoalsAttempted': "INTEGER", 'matchupFieldGoalPercentage': "FLOAT", 'matchupThreePointersMade': "INTEGER", 'matchupThreePointersAttempted': "INTEGER", 'matchupThreePointerPercentage': "FLOAT"}}]}, 'awayTeam': {'teamId': "INTEGER", 'players': [{'personId': "INTEGER", 'position': "STRING", 'statistics': {'matchupMinutes': "STRING", 'partialPossessions': "FLOAT", 'switchesOn': "INTEGER", 'playerPoints': "INTEGER", 'defensiveRebounds': "INTEGER", 'matchupAssists': "INTEGER", 'matchupTurnovers': "INTEGER", 'steals': "INTEGER", 'blocks': "INTEGER", 'matchupFieldGoalsMade': "INTEGER", 'matchupFieldGoalsAttempted': "INTEGER", 'matchupFieldGoalPercentage': "FLOAT", 'matchupThreePointersMade': "INTEGER", 'matchupThreePointersAttempted': "INTEGER", 'matchupThreePointerPercentage': "FLOAT"}}]}}
            },
            'PBP': {
                'bigquery_table': 'playbyplay',
                'description': 'Table with play-by-play data for each game',
                'endpoint': 'https://stats.nba.com/stats/playbyplayv3?',
                'version': 3,
                'parameters': {
                    'GameID': '0042300105', 
                    'StartPeriod': '1', 
                    'EndPeriod': '4'
                },
                'table_name': 'game',
                'fields': {'gameId': "INTEGER", 'actions': [{'actionNumber': "INTEGER", 'clock': "STRING", 'period': "INTEGER", 'teamId': "INTEGER", 'teamTricode': "STRING", 'personId': "INTEGER", 'playerName': "STRING", 'playerNameI': "STRING", 'xLegacy': "INTEGER", 'yLegacy': "INTEGER", 'shotDistance': "INTEGER", 'shotResult': "STRING", 'isFieldGoal': "INTEGER", 'scoreHome': "INTEGER", 'scoreAway': "INTEGER", 'pointsTotal': "INTEGER", 'location': "STRING", 'description': "STRING", 'actionType': "STRING", 'subType': "STRING", 'videoAvailable': "INTEGER", 'shotValue': "INTEGER", 'actionId': "INTEGER"}]}
            }
        }

        # Exisitng leagues in the API. The ID might be necessary as a parameter depending which endpoint is selected to run
        self.LEAGUE_IDS = {
            "00": "NBA",
            "10": "WNBA",
            "12": "Gaming",
            "13": "Summer League",
            "15": "Summer League",
            "16": "Summer League",
            "20": "G-League",
        }

        # Default headers for requests calls
        self.STATS_HEADERS = {
            "Host": "stats.nba.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
            "Connection": "keep-alive",
            "Referer": "https://stats.nba.com/",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }

        # print("An NBA Class instance was initiaited. A summary of the data that can be extracted from this object as well as its endpoints are displayed below")
        # print(json.dumps(self.VAR_NAMING_GUIDE, sort_keys=True, indent= 4))

        # Define game date. If not provided on the class call, set as date prior to run date
        if game_date is None:
            today = date.today()
            game_date = today - timedelta(days=1)
            self.GAME_DATE = datetime.strftime(game_date, '%Y-%m-%d')
        elif isinstance(game_date, date):
            self.GAME_DATE = datetime.strftime(game_date, '%Y-%m-%d')
        else:
            raise Exception("Please enter a date following the format of datetime.date(2024, 11, 9)")

        # Define game set. If not provided on the class call, gather the games from the game date defined above
        if game_set == {}:
            self.GAME_SET = set([row['GAME_ID'] for row in self.func_run_proc(self.VAR_NAMING_GUIDE['GAME_DATA'], params = {'GameDate': self.GAME_DATE, 'LeagueID': '00'})])
        elif isinstance(game_set, set):
            self.GAME_SET = game_set
        else:
            raise Exception("Please enter a SET {} of game IDs")

    def serialize(self) -> dict:
        return self.__dict__

    @staticmethod
    def deserialize(self):
        return NBA(self.game_date, self.game_set)

    def func_filter_data_to_skeleton(self, raw_data, skeleton):
        """
            Recursively filters raw data to match the structure of the skeleton.
            Used on version 3 of endpoints
        """
        if isinstance(skeleton, dict):
            # If the skeleton is a dictionary, filter keys to match the skeleton
            return {
                key: self.func_filter_data_to_skeleton(raw_data.get(key, None), value)
                for key, value in skeleton.items()
                #if key in raw_data  # Only include keys present in raw data
            }
        elif isinstance(skeleton, list):
            # If the skeleton is a list, map each item in the raw list to the skeleton's first element
            if len(skeleton) > 0 and isinstance(raw_data, list):
                if len(raw_data) > 0: # If key exists in raw data, extract key-value pair
                    return [
                        self.func_filter_data_to_skeleton(item, skeleton[0]) for item in raw_data
                    ]
                else: # If key is not present in raw data, force key to exist with None as value
                    return [{
                        key: None for key, value in skeleton[0].items()
                    }]
            else:
                return []
        else:
            # For primitive types (strings, numbers, etc.), return the raw data if it exists
            return raw_data if raw_data is not None else None

    def func_filter_results_by_list(self, results, headers, name):
        '''
            Reads list of lists structure that contains table headers and results.
            Transforms into a list of dictionaries ready to be loaded
            To be used on version 2 of endpoints
        '''

        # Get expected list. Some result lists may contain multiple outputs, so the expected list is set up within the __init__ function
        idx = [i['name'] for i in results].index(name)

        # Get the selected headers (__init__ function setup) positions within the list selected above
        headers_loc = [results[idx]['headers'].index(i) for i in headers]

        # Get the elements within each list (inside the result list) for the positions defined in the last step
        get_elements = operator.itemgetter(*headers_loc)

        # Combine the headers with the results into a JSON-like dictionary
        result = [dict(zip(headers,list(get_elements(row)))) for id, row in enumerate(results[idx]['rowSet'])]

        return result

    def func_load_data(self, data, table, schema = [], write_disposition = 'append'):
        '''
            Load data into BigQuery tables
            Process is preset to append data, taking dataframe into consideration
        '''

        # Create BigQuery client
        client = bigquery.Client()

        # Setup BigQuery load configurations
        job_config = bigquery.LoadJobConfig()

        # Set write disposition to append or overwrite data to BigQuery table
        if write_disposition == 'append':
            job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_APPEND
        elif write_disposition == 'overwrite':
            job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        # The source format defaults to CSV, so the line below is optional.
        # job_config.source_format = bigquery.SourceFormat.CSV

        job_config.schema = schema

        # Set project, database and table IDs for the load 
        project = client.project
        dataset_id = bigquery.DatasetReference(project, 'nba')
        table_id = dataset_id.table(table)

        # Run the load job
        job = client.load_table_from_json(data, table_id, job_config=job_config)  # Make an API request.
        job.result()  # Wait for the job to complete.

        print(f'Loaded table {table}')

    def func_run_proc(self, table, params = None):
        '''
            Run the process for a selected table
            Any and all endpoints defined in the __init__ function can be used in the function
        '''
        # Setup some initial configurations
        if not params: # All the endpoints need paramters to work correctly. If none is provided, default ones will be used
            print("Ensure params value is filled with a dictionary, otherwise, default value will apply")
            print(f"Applied value: {table['parameters']}")
            parameters = table['parameters']
        elif (not isinstance(params, dict)): # Parameters need to be provided as a dictionary to work correctly in the process, otherwise an error is returned
            raise Exception(f"Ensure params value is filled with a dictionary and that it contains the correct keys. Expected format: {table['parameters']}")
        elif (any(i not in table['parameters'].keys() for i in params.keys())): # If a non-expected key is provided as parameter and error is returned
            raise Exception(f"Ensure params value is filled with a dictionary and that it contains the correct keys. Expected format: {table['parameters']}")
        else: # If all parameters are provided as expected, they are assigned to the parameters variable
            parameters = params

        # Build the URL, with the correct endpoint and parameters and make the GET request
        url_params = "&".join([f"{key}={value}" for key, value in parameters.items()])
        url = f"{table['endpoint']}{url_params}"
        r = requests.get(url, headers = self.STATS_HEADERS)

        # Depending on the version of the endpoint (defined on the __init__ function), run the expected function
        if table['version'] == 3:
            final_data = self.func_filter_data_to_skeleton(r.json()[table['table_name']], table['fields'])
        elif table['version'] == 2:
            final_data = self.func_filter_results_by_list(r.json()['resultSets'], list(table['fields'].keys()), table['table_name'])
        
        return final_data
    
    def func_merge_lists_of_dicts(self, list1, list2, key):
        """
            Merges two lists of dictionaries based on a common key.
        """

        # Empty list to be used below
        merged_list = []

        # Go through each dictionary in list1, compared to each dictionary in list2 and combine them if they share the same value for a specific key
        for dict1 in list1:
            for dict2 in list2:
                if dict1[key] == dict2[key]:
                    merged_dict = {**dict1, **dict2}
                    merged_list.append(merged_dict)

        return merged_list

    def func_create_schema_from_skeleton(self, skeleton, parent_field=""):
        """
            Recursively creates a BigQuery schema from a skeleton.
            The skeleton is a dictionary representing the structure of the data.
        """
        schema = []
        # Iterate over the skeleton keys and types
        for key, value in skeleton.items():
            # Handle parent field (useful for nested fields)
            full_field_name = f"{parent_field}.{key}" if parent_field else key
            
            if isinstance(value, dict):
                # If the value is a dictionary, create a RECORD type field
                schema.append(bigquery.SchemaField(key, "RECORD", mode="NULLABLE", fields=self.func_create_schema_from_skeleton(value, full_field_name)))
            elif isinstance(value, list):
                # If the value is a list, determine if it is a repeated field
                if len(value) > 0 and isinstance(value[0], dict):
                    # If the first item in the list is a dictionary, it's a REPEATED RECORD
                    schema.append(bigquery.SchemaField(key, "RECORD", mode="REPEATED", fields=self.func_create_schema_from_skeleton(value[0], full_field_name)))
                else:
                    # Otherwise, treat it as a REPEATED type (e.g., REPEATED STRING, REPEATED INTEGER)
                    schema.append(bigquery.SchemaField(key, "STRING", mode="REPEATED"))  # Default to STRING for simplicity
            else:
                # Handle primitive types (STRING, INTEGER, etc.)
                schema.append(bigquery.SchemaField(key, value, mode="NULLABLE"))
        
        return schema    

    def func_run_all(self):
        '''
            Process to gather all basic box scores and load them to respective tables in BigQuery.
            This function basically runs the same steps 3 times, with small variations to account for endpoint expected parameters
        '''

        # Define table name
        table = "BS_SUMMARY"

        # Extract data for selected table for each game in the GAME_SET
        bs_sum = [self.func_run_proc(self.VAR_NAMING_GUIDE[table], params = {"GameID": game}) for game in self.GAME_SET]

        # Add a gameDate field to each result so table can be partitioned in BigQuery
        [game_sum.update({"gameDate": datetime.strftime(datetime.strptime(game_sum['gameEt'],'%Y-%m-%dT%H:%M:%SZ').date(),'%Y-%m-%d')}) for game_sum in bs_sum]

        # Create schema to be used when loading to BigQuery based on fields defined in the __init__ function
        field_skeleton = self.VAR_NAMING_GUIDE[table]['fields']
        field_skeleton.update({'gameDate': 'DATE'})
        schema = self.func_create_schema_from_skeleton(field_skeleton)

        # Load selected table to BigQuery
        self.func_load_data(bs_sum, self.VAR_NAMING_GUIDE[table]['bigquery_table'], schema)

        # Extract gameDate for each game. It will be used later to gameDate to all result dictionaries
        game_dates = [{"gameId": game_sum["gameId"], "gameDate": game_sum["gameDate"]} for game_sum in bs_sum]

        # Define list of table names to loop through
        tables = ['BS_TRAD', 'BS_ADV', 'BS_MISC', 'BS_SCORE', 'BS_USAGE', 'BS_FOUR', 'BS_TRACK', 'BS_HUSTLE', 'BS_MATCH', 'BS_DEF']

        for table in tables:
            # Extract data for selected table for each game in the GAME_SET
            bs = [self.func_run_proc(self.VAR_NAMING_GUIDE[table], params = {"GameID": game}) for game in self.GAME_SET]
            
            # Add a gameDate field to each result (using the game_dates variable defined above) so table can be partitioned in BigQuery
            result = self.func_merge_lists_of_dicts(bs, game_dates, 'gameId')
            
            # Create schema to be used when loading to BigQuery based on fields defined in the __init__ function
            field_skeleton = self.VAR_NAMING_GUIDE[table]['fields']
            field_skeleton.update({'gameDate': 'DATE'})
            schema = self.func_create_schema_from_skeleton(field_skeleton)

            # Load selected table to BigQuery
            self.func_load_data(result, self.VAR_NAMING_GUIDE[table]['bigquery_table'], schema)

        # Define table name
        table = 'PBP'

        # Extract data for selected table for each game in the GAME_SET
        bs = [self.func_run_proc(self.VAR_NAMING_GUIDE[table], params = {"GameID": game, 'StartPeriod': '1', 'EndPeriod': '4'}) for game in self.GAME_SET]
        
        # Add a gameDate field to each result (using the game_dates variable defined above) so table can be partitioned in BigQuery
        result = self.func_merge_lists_of_dicts(bs, game_dates, 'gameId')     
        
        # Create schema to be used when loading to BigQuery based on fields defined in the __init__ function
        field_skeleton = self.VAR_NAMING_GUIDE[table]['fields']
        field_skeleton.update({'gameDate': 'DATE'})
        schema = self.func_create_schema_from_skeleton(field_skeleton)

        # Load selected table to BigQuery
        self.func_load_data(result, self.VAR_NAMING_GUIDE[table]['bigquery_table'], schema)

    
# %%

nba = NBA()

# %%

#################################################################
## Start NBA class with a selected set of games
#################################################################
 
# nba_test_2 = NBA(game_set = {'0022201073', '0022400231'})


#################################################################
## Start NBA class with a selected date
#################################################################

# nba_test_3 = NBA(game_date = date(2022,2,1))


# %%

#################################################################
## Games between two dates
#################################################################

## Start class
# nba_2 = NBA()

## Create a date list between two dates
# sdate = date(2024,12,16)   # start date
# edate = date(2024,12,20)   # end date
# dates = pd.date_range(sdate,edate,freq='d')


## Loop through dates, starting process for each date and running all processes
# for date_s in dates:
#     print(date_s)
#     nba_2 = NBA(game_date=date_s)
#     nba_2.func_run_all()


## Extract the API response that contain the IDs for games between those dates
# games_list = [nba_2.func_run_proc(nba_2.VAR_NAMING_GUIDE['GAME_DATA'], params = {'GameDate': date_s, 'LeagueID': '00'})
#     for date_s in dates]

## Save the IDs in the GAME_SET variable for the class
# nba_2.GAME_SET = list(itertools.chain(*[
#     [i['GAME_ID'] for i in dt]
#         for dt in games_list 
#     ]
# ))

## Run full process for the selected games
# nba_2.func_run_all()

# %%

#################################################################
## Games for a full past season
#################################################################

## Start class
# nba_3 = NBA()

## Create a date list between two dates
# sdate = date(2024,10,22)   # start date
# edate = date(2024,12,13)   # end date
# dates = pd.date_range(sdate,edate,freq='d')

## Loop through dates, starting process for each date and running all processes
# for date_s in dates:
#     print(date_s)
#     nba_3 = NBA(game_date=date_s)
#     nba_3.func_run_all()

## Extract the API response that contain the IDs for games between those dates
# games_list = [nba_3.func_run_proc(nba_3.VAR_NAMING_GUIDE['GAME_DATA'], params = {'GameDate': date_s, 'LeagueID': '00'})
#     for date_s in dates]

## Save the IDs in the GAME_SET variable for the class
# nba_3.GAME_SET = list(itertools.chain(*[
#     [i['GAME_ID'] for i in dt]
#         for dt in games_list 
#     ]
# ))

## Run full process for the selected games
# nba_3.func_run_all()

# %%

#################################################################
## All Players in history (high-level data)
#################################################################

# nba_4 = NBA()

# table = 'PLAYERS'

# player_df = nba_4.func_run_proc(nba_4.VAR_NAMING_GUIDE[table], params = {"LeagueID": "00"})

# field_skeleton = nba_4.VAR_NAMING_GUIDE[table]['fields']
# schema = nba_4.func_create_schema_from_skeleton(field_skeleton)

# nba_4.func_load_data(player_df, nba_4.VAR_NAMING_GUIDE[table]['bigquery_table'], schema, 'overwrite')

# %%

