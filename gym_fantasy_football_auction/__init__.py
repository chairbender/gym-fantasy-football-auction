from gym.envs.registration import register
import pkg_resources
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot

from gym_fantasy_football_auction.envs.agents import SimpleScriptedFantasyFootballAgent

PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.envs', 'data/cheatsheet.csv')
players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)

register(
    id='FFEnv1-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=1.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(30, 0.9, 0.1),
                      SimpleScriptedFantasyFootballAgent(30, 0.9, 0.1),
                      SimpleScriptedFantasyFootballAgent(30, 0.9, 0.1)],
        'players': players, 'money': 30,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.RB],
        'starter_value': 1,
        'reward_function': '3'
    }
)
