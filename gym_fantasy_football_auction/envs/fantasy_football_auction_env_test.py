"""
This defines the fantasy football task environment.

"""
import random
from unittest import TestCase
import pkg_resources
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot
from gym_fantasy_football_auction.envs import FantasyFootballAuctionEnv
from gym_fantasy_football_auction.envs.agents import SimpleScriptedFantasyFootballAgent

PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.envs', 'data/cheatsheet.csv')

def test_with_seed(seed):
    random.seed(seed)
    players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)
    # create a mock game with 4 of this agent and use this agent in it, verify that the game completes
    me = SimpleScriptedFantasyFootballAgent()
    opponents = [SimpleScriptedFantasyFootballAgent(), SimpleScriptedFantasyFootballAgent(),
                 SimpleScriptedFantasyFootballAgent()]

    env = FantasyFootballAuctionEnv(opponents,
                                    players,
                                    200,
                                    [RosterSlot.QB, RosterSlot.WR, RosterSlot.WR, RosterSlot.RB, RosterSlot.RB,
                                     RosterSlot.TE, RosterSlot.WRRBTE, RosterSlot.K, RosterSlot.DST,
                                     RosterSlot.BN, RosterSlot.BN],
                                    0.8)
    turncount = 0
    try:
        while True:
            observation, reward, done, info = env.step(me.act(env.auction, 0))
            turncount += 1
            if done:
                if info['error'] is not None:
                    raise info['error']
                break
    except Exception as err:
        print("On turn count " + str(turncount))
        print(env.auction)
        raise err




class FantasyFootballAuctionEnvTestCase(TestCase):
    def test_env_with_simple_agent(self):
        test_with_seed(123)
        test_with_seed(456)
        test_with_seed(789)


