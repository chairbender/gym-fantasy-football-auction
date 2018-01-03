"""
This defines agents which can play in fantasy football auction
"""
import abc
import random

from fantasy_football_auction.auction import AuctionState

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
    A simple scripted agent. Target bids at within a random distance of
    the value of each player.
    """

    def __init__(self):
        self.target = 0
        self.target_player = None

    def reset(self):
        self.target = 0
        self.target_player = None

    def act(self, auction, my_idx):
        owner_me = auction.owners[my_idx]
        nominee_index = auction.nominee_index()

        if auction.state == AuctionState.NOMINATE \
            and auction.turn_index == my_idx:
            # find the next player that we can buy (for at least the minimum)
            for player in auction.undrafted_players:
                if owner_me.can_buy(player, 1):
                    # nominate at some percentage of
                    # the real value, start at half
                    percent = random.uniform(0.8, 1.2)
                    self.target = int(round(min(max(percent * player.value, 1), owner_me.max_bid())))
                    self.target_player = player
                    return FantasyFootballAuctionEnv.action_index(auction, auction.players.index(player),
                                                                  max(1, int(round(self.target / 2.0))))
        elif auction.state == AuctionState.BID:
            if self.target_player != auction.nominee and owner_me.can_buy(auction.nominee, 1):
                # new target needs to be set if we can draft this player (for at least the min amount)
                percent = random.uniform(0.8, 1.2)
                self.target = int(round(percent * auction.nominee.value))
                self.target_player = auction.nominee

            # walk up to the target if it is not yet exceeded, if we aren't the current winner,
            # and if we can still bid that amount
            if auction.bid < self.target and auction.winning_owner_index() != my_idx:
                increment = int(round(random.uniform(1, self.target - auction.bid)))
                if owner_me.can_buy(auction.nominee, auction.bid + increment):
                    return FantasyFootballAuctionEnv.action_index(auction, nominee_index, auction.bid + increment)

        return 0


