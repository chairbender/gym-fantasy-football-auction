from gym.envs.registration import register
import pkg_resources
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot

from gym_fantasy_football_auction.envs.agents import SimpleScriptedFantasyFootballAgent

PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.envs', 'data/cheatsheet.csv')
players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)

register(
    id='FantasyFootballAuction-2OwnerSmallRoster-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=1.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.RB],
        'starter_value': .9
    }
)
