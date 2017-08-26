from gym.envs.registration import register
import pkg_resources
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot

PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.env', 'data/cheatsheet.csv')
players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)

register(
    id='FantasyFootballAuction-SimpleVsRandom',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    owner_idx=0, num_owners=2,players=players,money=200,
    roster=[RosterSlot.QB, RosterSlot.WR, RosterSlot.RB], opponent='random',
    starter_value=.9
)
