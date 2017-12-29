"""
This defines the fantasy football task environment.

TODO: Fix this so it works and add unit tests.
TODO: Fix how the observation and action space is set up based on README
TODO: Define FantasyFootballAgent and an implementation (simple scripted agent).
"""

import gym
import sys
from gym import error, spaces, utils
from gym.utils import seeding
from fantasy_football_auction.auction import Auction, AuctionState
from fantasy_football_auction.position import Position, RosterSlot
from six import StringIO


class FantasyFootballAuctionEnv(gym.Env):
    """
    Fantasy football auction draft, with values for each draftable player pre-determined.

    The agent is always assumed to be the first (owner index 0) owner.
    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, opponents, players, money, roster, starter_value):
        """
        :param list(FantasyFootballAgent) opponents: the other opponents in this game
        :param list(Player) players: list of fantasy football auction Players that will be drafted
        :param int money: how much money each owner in the auction starts with
        :param list(RosterSlot) roster: the slots each owner must fill
        :param starter_value: floating point between 0 and 1 inclusive indicating how heavily the final score should
            be weighted between starter and bench. If 1, for example, bench value will be ignored when calculating
            winners.
        """
        self.opponents = opponents
        self.players = players
        self.money = money
        self.roster = roster
        self.starter_value = starter_value

        self.auction = Auction(players, len(opponents), money, roster)
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.done = False

        self._seed()

    def _seed(self, seed=None):
        # Used to seed the random opponent
        self.np_random, seed1 = seeding.np_random(seed)
        return [seed1]

    def _action_space(self):
        """
        See README.md for details

         We represent the action space as a 2-dimensional matrix. The row
        represents the player index. The column represents the bid on that
        player - one column for every integer from 0 to the maximum possible
        bid amount.

        This is represented as a MultiDiscrete with the first dimension as the player
        index and the second as the money

        :return: the action space
        """

        return spaces.MultiDiscrete([[0, len(self.players)], [0, self.money]])

    def _observation_space(self):
        """
         See README.md for details.

        This returns a multidiscrete with the following dimensions (in this order).

        Say we have p draftable players, m starting money, and
        n owners in the game.
        We have these dimensions (in the order described):
        n dimensions in the range (0 - m): the current max bid for each owner
        1 dimension in the range (0 - m): the current bid to beat for the current nominee
        1 dimension in the range (0 - n): the index of the owner who has the current bid to beat
            for the current nominee
        p dimensions in the range (0 - n+1):
          * each dimension represents a draftable player's 'state'
          * a value of 0 - n-1 indicates that the owner with that index owns that player
          * a value of n indicates that the player is undrafted
          * a value of n+1 indicates that the player is the current nominee

        Each of these is represented in a multidiscrete with the dimensions in the order
        specified above.

        :return: the observation space
        """
        dimensions = []
        # one per owner - max bid
        for i in range(len(self.opponents)):
            dimensions.append([0, self.money])

        # bid to beat
        dimensions.append([0, self.money])
        # winning bidder
        dimensions.append([0, len(self.opponents)])

        # player status, one per player
        for i in range(len(self.players)):
            dimensions.append([0, len(self.opponents)+1])

        return spaces.MultiDiscrete(dimensions)

    def _playerOwnersPurchases(self):
        """

        :return: an array with each element corrresponding to each player in self.players indicating the owner
        of the player and the purchase of that player. None indicates no owner. The element is a dictionary with an owner and a purchase
        """
        result = [None] * len(self.players)
        for owner in self.auction.owners:
            for purchase in owner.purchases:
                player_index = self.players.index(purchase.player)
                result[player_index] = {owner: owner, purchase: purchase}

        return result

    def _encode_auction(self):
        """

        Array with the value of each dimension:

        We have these dimensions (in the order described):
        n dimensions in the range (0 - m): the current max bid for each owner
        1 dimension in the range (0 - m): the current bid to beat for the current nominee
        1 dimension in the range (0 - n): the index of the owner who has the current bid to beat
            for the current nominee
        p dimensions in the range (0 - n+1):
          * each dimension represents a draftable player's 'state'
          * a value of 0 - n-1 indicates that the owner with that index owns that player
          * a value of n indicates that the player is undrafted
          * a value of n+1 indicates that the player is the current nominee

        :return: the observation space of the current auction
        """

        observation = []

        # one per owner - max bid
        for owner in self.auction.owners:
            observation.append([owner.max_bid()])

        # bid to beat
        observation.append(self.auction.bid)
        max_bid = -1
        for bid in enumerate(self.auction.bids):
            max_bid = max(max_bid, bid)
        max_bid_idx = -1
        for i, bid in enumerate(self.auction.bids):
            if max_bid == bid:
                max_bid_idx = i

        # winning bidder
        observation.append(max_bid_idx)

        # player status, one per player
        for player in self.auction.players:
            owner_idx = -1
            for i, owner in enumerate(self.auction.owners):
                if owner.owns(player):
                    owner_idx = i
            if owner_idx != -1:
                observation.append(owner_idx)
            else:
                if player == self.auction.nominee:
                    observation.append(len(self.auction.owners) + 1)
                else:
                    observation.append(len(self.auction.owners))

        return observation


    def _reset(self):
        self.auction = Auction(self.players, self.num_owners, self.money, self.roster)
        self.done = False

        self._reset_opponent()

        return self._encode_auction()

    def _close(self):
        self.owner_idx = None
        self.players = None
        self.num_owners = None
        self.money = None
        self.roster = None
        self.auction = None
        self.action_space = None
        self.observation_space = None
        self.done = None
        self.opponent_policies = None
        self.opponent = None
        self.starter_value = None
        self.opponents = None


    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.auction) + '\n')
        return outfile

    def _act(self, action, owner_idx):
        """

        :param action: action to perform, should be an array with 2 values, the first being the
            player to nominate (0 to not nominate), the second being the bid amount (0 to not bid)
        :param owner_idx: index of owner who should do the action
        :return: true if success, false if illegal move
        """
        if action[0] > 0:
            return self.auction.nominate(owner_idx,action[0]-1,action[1])
        elif action[1] > 0:
            return self.auction.place_bid(owner_idx,action[1])

    def _step(self, action):
        """

        :param action: action state to choose within action space
        :return tuple:
            (observation, reward, done, info)
            observation is a tuple representing the current observation state within observation space.
            reward is a float representing the reward achieved by the previous action
            done is a boolean indicating whether the task is done and it's time to restart
            info is a diagnostic tool for debugging
        """
        # If already terminal, then don't do anything
        if self.done:
            return self._encode_auction(), 0., True, {}

        if not self._act(action, self.owner_idx):
            # Automatic loss on illegal move
            self.done = True
            return self._encode_auction(), -1., True, {}

        # All opponents play
        if not self.auction.state == AuctionState.DONE:
            for policy in self.opponent_policies:
                policy(self.auction)

        self.auction.tick()

        # Reward: if nonterminal, then the reward is 0
        if self.auction.state != AuctionState.DONE:
            self.done = False
            return self._encode_auction(), 0., False, {}
        else:
            # We're in a terminal state. Reward is a gradient between -1 and 1 depending on standing
            self.done = True
            scores = self.auction.scores(self.starter_value)
            range = (min(scores),max(scores))
            my_score = scores[self.owner_idx]
            distance = (my_score - range[0]) / (range[1] - range[0])
            reward = ((distance * 2) - 1) ** 2
            return self.state.board.encode(), reward, True, {}

    def _reset_opponent(self):
        if self.opponent == 'random':
            # generate a policy for every player but the agent
            self.opponent_policies = \
                [make_random_policy(self.np_random,i) for i in
                 filter(lambda j: j != self.owner_idx, range(0, self.num_owners))]
        else:
            raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))