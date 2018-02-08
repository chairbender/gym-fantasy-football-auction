"""
This defines agents which can play in fantasy football auction
"""
import abc
import random

from fantasy_football_auction.auction import AuctionState
import numpy as np

from gym_fantasy_football_auction.envs import FantasyFootballAuctionEnv


class FantasyFootballAgent:
    """
    Interface for being an Agent that works with the fantasy
    football auction environment.
    """
    @abc.abstractmethod
    def act(self, auction, my_idx):
        """

        :param Auction auction: current state of the auction
        :param int my_idx: the owner index of this agent
        :return int: integer representing the index of the action to choose from
            within the flattened action space
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        invoked when agent should be reset to initial state
        """
        pass


class SimpleScriptedFantasyFootballAgent(FantasyFootballAgent):
    """
    A simple scripted agent.
    First, it calculates a valuation for each player.
    The formula for each is:
    fraction = randomly pick (1-inaccuracy,1+inaccuracy)
    fraction = sample from normal with mean=fraction std_dev=std_dev
    actual * fraction * (money / 200)

    (the 200 is because each player's listed value is based on a 200 dollar auction, but we don't always
    have the environment use 200 dollar starting budget so it needs to be scaled down)

    It always nominates whoever it thinks is the next highest value player, given by its valuation formula. Starts
    bid at half their valuation.

    During bidding, it then randomly raises the value each round up to the valuation, then drops out.

    Attributes:
        :ivar int money: float. Total money. Used to calculate valuation.
        Player values are assumed to be based on a 200 dollar auction.
        So actual values will be recalculated based on the ratio.
        :ivar float accuracy: see above for valuation formula
        :ivar float std_dev: see above for valuation formula
    """

    def __init__(self, money=200, inaccuracy=0.5, std_dev=0.05):
        self.target = 0
        self.target_player = None
        self.inaccuracy = inaccuracy
        self.money = money
        self.valuation_ratio = money / 200
        self.std_dev = std_dev

    def reset(self):
        self.target = 0
        self.target_player = None

    def valuation(self, player):
        """

        :param player: player to valuate
        :return: valuation using the valuation formula
        """
        fraction = random.choice([1-self.inaccuracy,1+self.inaccuracy])
        fraction = random.normalvariate(fraction, self.std_dev)

        return player.value * fraction * self.valuation_ratio

    def act(self, auction, my_idx):
        owner_me = auction.owners[my_idx]
        nominee_index = auction.nominee_index()

        if auction.state == AuctionState.NOMINATE \
            and auction.turn_index == my_idx:
            # valuate all draftable players then pick the max

            valuations = [self.valuation(player) if owner_me.can_buy(player, 1) else -1 for player in auction.undrafted_players]
            chosen_index = np.argmax(valuations)
            chosen_player = auction.undrafted_players[chosen_index]

            self.target = int(round(min(max(valuations[chosen_index], 1), owner_me.max_bid())))
            self.target_player = chosen_player
            return FantasyFootballAuctionEnv.action_index(auction, auction.players.index(chosen_player),
                                                          max(1, int(round(self.target / 2.0))))
        elif auction.state == AuctionState.BID:
            if self.target_player != auction.nominee and owner_me.can_buy(auction.nominee, 1):
                self.target = int(round(self.valuation(auction.nominee)))
                self.target_player = auction.nominee

            # walk up to the target if it is not yet exceeded, if we aren't the current winner,
            # and if we can still bid that amount
            if auction.bid < self.target and auction.winning_owner_index() != my_idx:
                increment = int(round(random.uniform(1, self.target - auction.bid)))
                if owner_me.can_buy(auction.nominee, auction.bid + increment):
                    return FantasyFootballAuctionEnv.action_index(auction, nominee_index, auction.bid + increment)

        return 0


