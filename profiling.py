"""
For investigating performance issues
"""
import time
import random
import cProfile
import pkg_resources
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot
from gym_fantasy_football_auction.envs import FantasyFootballAuctionEnv
from gym_fantasy_football_auction.envs.agents import SimpleScriptedFantasyFootballAgent

PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.envs', 'data/cheatsheet.csv')
players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)

def test_with_seed(seed):
    random.seed(seed)

    # create a mock game with 6 of this agent and use this agent in it, verify that the game completes
    me = SimpleScriptedFantasyFootballAgent()
    opponents = [SimpleScriptedFantasyFootballAgent(), SimpleScriptedFantasyFootballAgent(),
                 SimpleScriptedFantasyFootballAgent(), SimpleScriptedFantasyFootballAgent(),
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
        print("Error on turn count " + str(turncount))
        print("With seed " + str(seed))
        print(env.auction)
        raise err


# run 100 random games and see how it performs
def perf_test():
    try:
        i = 0
        for i in range(100):
            test_with_seed(int(round(time.time())))
    except Exception as err:
        print("error on test iteration " + str(i))
        raise err

cProfile.run('perf_test()')
