"""
This defines the fantasy football task environment.

"""
import random
from unittest import TestCase

import gym
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
    me = SimpleScriptedFantasyFootballAgent(200, 0.9, 0.1)
    opponents = [SimpleScriptedFantasyFootballAgent(200, 0.9, 0.1), SimpleScriptedFantasyFootballAgent(200, 0.9, 0.1),
                 SimpleScriptedFantasyFootballAgent(200, 0.9, 0.1), SimpleScriptedFantasyFootballAgent(200, 0.9, 0.1),
                 SimpleScriptedFantasyFootballAgent(200, 0.9, 0.1)]

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
            if env.error is not None:
                raise env.error
            if done:
                break
    except Exception as err:
        print("On turn count " + str(turncount))
        print(env.auction)
        raise err

def test_gym_env_with_seed(env, seed):
    env.reset()
    random.seed(seed)
    # create a mock game with 6 of this agent and use this agent in it, verify that the game completes
    me = SimpleScriptedFantasyFootballAgent()

    turncount = 0
    try:
        while True:
            observation, reward, done, info = env.step(me.act(env.auction, 0))
            turncount += 1
            if env.error is not None:
                raise env.error
            if done:
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
        test_with_seed(1514540459)

    def test_easy_env_with_gym(self):
        env = gym.make('FFEnv1-v0')
        test_gym_env_with_seed(env, 111)
        test_gym_env_with_seed(env, 222)
        test_gym_env_with_seed(env, 333)

        env.reset()
        test_gym_env_with_seed(env, 111)
        test_gym_env_with_seed(env, 222)
        test_gym_env_with_seed(env, 333)

        env.reset()
        test_gym_env_with_seed(env, 111)
        test_gym_env_with_seed(env, 222)
        test_gym_env_with_seed(env, 333)



