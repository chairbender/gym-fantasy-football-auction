from gym.envs.registration import register
import pkg_resources
from fantasy_football_auction.player import players_from_fantasypros_cheatsheet
from fantasy_football_auction.position import RosterSlot

from gym_fantasy_football_auction.envs.agents import SimpleScriptedFantasyFootballAgent

PLAYERS_CSV_PATH = pkg_resources.resource_filename('gym_fantasy_football_auction.envs', 'data/cheatsheet.csv')
players = players_from_fantasypros_cheatsheet(PLAYERS_CSV_PATH)

register(
    id='FantasyFootballAuction-2OwnerSingleRosterSimpleScriptedOpponent-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=1.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB],
        'starter_value': 1,
        'reward_function': '2'
    }
)

register(
    id='FantasyFootballAuction-2OwnerSmallRosterSimpleScriptedOpponent-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=1.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.RB],
        'starter_value': 1,
        'reward_function': '3'
    }
)

register(
    id='FantasyFootballAuction-4OwnerSmallRosterSimpleScriptedOpponent-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=5.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.RB],
        'starter_value': 1,
        'reward_function': '3'
    }
)

register(
    id='FantasyFootballAuction-4OwnerMediumRosterSimpleScriptedOpponent-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=10.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.RB, RosterSlot.TE, RosterSlot.WRRBTE],
        'starter_value': 1,
        'reward_function': '3'
    }
)

register(
    id='FantasyFootballAuction-6OwnerMediumRosterSimpleScriptedOpponent-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=20.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.RB, RosterSlot.TE, RosterSlot.WRRBTE],
        'starter_value': 1,
        'reward_function': '3'
    }
)


register(
    id='FantasyFootballAuction-4OwnerFullRosterSimpleScriptedOpponent-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=20.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.WR,
                   RosterSlot.RB, RosterSlot.RB, RosterSlot.TE, RosterSlot.WRRBTE, RosterSlot.K, RosterSlot.DST,
                   RosterSlot.BN, RosterSlot.BN, RosterSlot.BN, RosterSlot.BN, RosterSlot.BN, RosterSlot.BN],
        'starter_value': .9,
        'reward_function': '3'
    }
)

register(
    id='FantasyFootballAuction-6OwnerFullRosterSimpleScriptedOpponent-v0',
    entry_point='gym_fantasy_football_auction.envs:FantasyFootballAuctionEnv',
    reward_threshold=100.0,

    kwargs={
        'opponents': [SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent(),
                      SimpleScriptedFantasyFootballAgent()],
        'players': players, 'money': 200,
        'roster': [RosterSlot.QB, RosterSlot.WR, RosterSlot.WR,
                   RosterSlot.RB, RosterSlot.RB, RosterSlot.TE, RosterSlot.WRRBTE, RosterSlot.K, RosterSlot.DST,
                   RosterSlot.BN, RosterSlot.BN, RosterSlot.BN, RosterSlot.BN, RosterSlot.BN, RosterSlot.BN],
        'starter_value': .9,
        'reward_function': '3'
    }
)
